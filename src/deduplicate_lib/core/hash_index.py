"""On-disk hash tables for MultiHashing (``index_dir``).

Layout::

    index_dir/metadata.json        parameters, dataset fingerprint, current files
    index_dir/counts.g<N>.bin      uint32 unique counts (raw little-endian), one per
                                   indexed vector; may hold more, only the first
                                   ``n_vectors`` are committed
    index_dir/p<PPPP>.g<N>.npy     sorted distinct uint64 hashes of perturbation P
    index_dir/p<PPPP>.g<N>.fence.npy  its fence: the first hash of each file page

Each perturbation has a base table (``hash_files``, written by the build) and a
list of smaller **delta segments** (``delta_files``, oldest first), written by
appends.  A segment holds only hashes found in no other segment of the same
perturbation, so a hash is in the table if it is in any of them.  Appending
writes the new hashes as a new delta instead of rewriting the whole table, and
segments of similar size are merged (see ``MultiHashing._save_index``), so an
append costs about the same however big the index is.

Segments are loaded as memory-mapped :class:`SortedSegment` s with their fence
(a small sidecar file, recomputed if missing) in memory, so a lookup in a table
larger than the page cache reads one page.

A vector's count is final once it is added, so a commit that only appends
vectors appends their counts to the current counts file (after cutting off
anything an interrupted commit left beyond ``n_vectors``) instead of rewriting
it; commits that change earlier counts (the build) write a new file.

``metadata.json`` is the commit point: it is replaced atomically and names the
current files of every table.  Table and count files are only ever written
under new generation numbers, so a crash leaves the previous commit intact;
files no commit references are deleted the next time the index is opened.

A single process should write to an index at a time.
"""

import contextlib
import hashlib
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from deduplicate_lib.core.sorted_segment import SortedSegment
from deduplicate_lib.core.vector_store import VectorStore

logger = logging.getLogger(__name__)

FORMAT_VERSION = 3
# Older formats this version opens (and upgrades on the next commit): format 2
# has no delta segments.
READABLE_FORMAT_VERSIONS = (2, 3)
# fsync every file and the directory on commit, so a commit survives a power
# loss (not just a killed process). Costly on some filesystems; tests disable it.
FSYNC = True
# Files fsynced concurrently on commit: filesystems batch concurrent fsyncs
# (3.5x faster than one at a time on btrfs with compression).
FSYNC_THREADS = 16
METADATA_FILE = "metadata.json"
# The dataset fingerprint is one sha256 per block of this many rows, so growing
# the index re-hashes only the last (partial) block and the new ones.
FINGERPRINT_BLOCK_ROWS = 2**20
# Bytes read at a time while hashing.
_FINGERPRINT_READ_BYTES = 2**26
_DATA_FILE = re.compile(r"^(p\d{4,}\.g\d+(\.fence)?\.npy|counts\.g\d+\.(npy|bin))$")
_COUNTS_DTYPE = np.dtype("<u4")
_TMP_FILE = re.compile(r"^.*\.tmp$")

# Parameters that must match for an index to be reused, with a hint each.
_MATCHED_PARAMETERS = {
    "format_version": "the index was written by another deduplicate_lib version",
    "hash_version": "the index was written by another deduplicate_lib version",
    "tolerance": "each tolerance needs its own index_dir",
    "perturbations": "",
    "perturbation_sha256": "the perturbation array (seed / pertrubation_array) differs",
    "dim": "",
}


class IndexMismatchError(ValueError):
    """An index directory does not belong to the current dataset or parameters."""


def fence_name(segment_name: str) -> str:
    """Name of the fence sidecar of a segment file."""
    return segment_name.removesuffix(".npy") + ".fence.npy"


def perturbation_digest(perturbation_array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(perturbation_array, dtype=np.float64).tobytes()
    ).hexdigest()


def dataset_fingerprint(
    store: VectorStore, n_vectors: int, previous: dict | None = None
) -> dict:
    """sha256 digests of every ``FINGERPRINT_BLOCK_ROWS``-row block of the first
    ``n_vectors`` rows, so any change to those rows changes the fingerprint.

    Values are hashed as float64, so the same data in another dtype (e.g. a
    float32 file loaded into a float64 array) has the same fingerprint.  With
    ``previous`` (the fingerprint of a shorter prefix of the same data), its
    complete blocks are reused rather than read again.
    """
    block_rows = FINGERPRINT_BLOCK_ROWS
    blocks: list[str] = []
    if (
        previous is not None
        and previous.get("block_rows") == block_rows
        and previous.get("n_vectors", n_vectors + 1) <= n_vectors
    ):
        blocks = list(previous["blocks"][: previous["n_vectors"] // block_rows])
    read_rows = max(1, _FINGERPRINT_READ_BYTES // (8 * max(1, store.dim)))
    for block_start in range(len(blocks) * block_rows, n_vectors, block_rows):
        block_stop = min(block_start + block_rows, n_vectors)
        digest = hashlib.sha256()
        for _, rows in store.iter_chunks(read_rows, block_start, block_stop):
            digest.update(np.ascontiguousarray(rows, dtype=np.float64).tobytes())
        blocks.append(digest.hexdigest())
    return {"n_vectors": n_vectors, "block_rows": block_rows, "blocks": blocks}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fsync_dir(path: Path) -> None:
    if not FSYNC:
        return
    with contextlib.suppress(OSError):  # not supported everywhere (e.g. Windows)
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class HashIndex:
    """An index directory: open (validating it) or create it with :meth:`open`."""

    def __init__(self, path: Path, metadata: dict) -> None:
        self.path = path
        self.metadata = metadata
        self._staged: dict[int, str] = {}
        # perturbation -> its full list of delta files after the next commit
        self._staged_deltas: dict[int, list[str]] = {}
        self._generation = metadata["generation"]
        if "delta_files" not in self.metadata:  # format 2
            self.metadata["delta_files"] = [[] for _ in self.metadata["hash_files"]]

    # --- opening ---------------------------------------------------------

    @classmethod
    def open(
        cls, path: str | Path, parameters: dict, store: VectorStore
    ) -> "HashIndex":
        """Open the index at ``path`` for ``store``, creating it if the directory is new.

        ``parameters`` holds the values in ``_MATCHED_PARAMETERS`` (plus any
        informational extras such as ``seed``).  Raises
        :class:`IndexMismatchError` if an existing index was built with other
        parameters or from another dataset.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        metadata_path = path / METADATA_FILE

        if not metadata_path.exists():
            foreign = [
                f.name
                for f in path.iterdir()
                if not (_DATA_FILE.match(f.name) or _TMP_FILE.match(f.name))
            ]
            if foreign:
                raise IndexMismatchError(
                    f"{path} is not empty and is not a deduplicate_lib index "
                    f"(found {sorted(foreign)[:5]}); choose a new index_dir."
                )
            index = cls._create(path, parameters, store)
            logger.info("Created hash index %s for %d vectors.", path, len(store))
        else:
            metadata = json.loads(metadata_path.read_text())
            index = cls(path, metadata)
            index._validate(parameters, store)
            logger.info(
                "Opened hash index %s: %d vectors, %d/%d perturbations done.",
                path,
                index.n_vectors,
                len(index.completed),
                parameters["perturbations"],
            )
        index._remove_unreferenced_files()
        return index

    @classmethod
    def _create(cls, path: Path, parameters: dict, store: VectorStore) -> "HashIndex":
        n_vectors = len(store)
        metadata = {
            **parameters,
            "format_version": FORMAT_VERSION,
            "n_vectors": n_vectors,
            "fingerprint": dataset_fingerprint(store, n_vectors),
            "source": repr(store),
            "hash_files": [None] * parameters["perturbations"],
            "delta_files": [[] for _ in range(parameters["perturbations"])],
            "counts_file": None,
            "generation": 0,
            "created": _now(),
            "updated": _now(),
        }
        index = cls(path, metadata)
        index._write_metadata()
        return index

    def _validate(self, parameters: dict, store: VectorStore) -> None:
        expected = {**parameters, "format_version": FORMAT_VERSION}
        problems = []
        for key, hint in _MATCHED_PARAMETERS.items():
            if (
                key == "format_version"
                and self.metadata.get(key) in READABLE_FORMAT_VERSIONS
            ):
                continue
            if self.metadata.get(key) != expected.get(key):
                line = f"{key}: index has {self.metadata.get(key)!r}, now {expected.get(key)!r}"
                problems.append(f"{line} ({hint})" if hint else line)
        if problems:
            raise IndexMismatchError(
                f"The hash index in {self.path} was built with different parameters:\n  "
                + "\n  ".join(problems)
                + "\nUse another index_dir (or delete this one) to build a new index."
            )

        if len(store) < self.n_vectors:
            raise IndexMismatchError(
                f"The hash index in {self.path} covers {self.n_vectors} vectors but the "
                f"dataset has only {len(store)}; it was built from another dataset."
            )
        if dataset_fingerprint(store, self.n_vectors) != self.metadata["fingerprint"]:
            raise IndexMismatchError(
                f"The dataset differs from the one the hash index in {self.path} was "
                f"built from (built from {self.metadata.get('source')}). Use another "
                "index_dir (or delete this one) to build a new index."
            )

    # --- reading ---------------------------------------------------------

    @property
    def n_vectors(self) -> int:
        """Number of leading dataset vectors the committed tables cover."""
        return self.metadata["n_vectors"]

    @property
    def completed(self) -> list[int]:
        """Perturbations whose tables are committed."""
        return [p for p, name in enumerate(self.metadata["hash_files"]) if name]

    def _load_segment(self, name: str) -> SortedSegment:
        table = np.load(self.path / name, mmap_mode="r")
        fence_path = self.path / fence_name(name)
        fence = None
        if fence_path.exists():
            fence = np.load(fence_path)
        else:
            logger.info("Computing the missing fence of %s.", name)
        return SortedSegment(table, fence)

    def load_table(self, perturbation: int) -> SortedSegment:
        """The current table of ``perturbation``, memory-mapped (staged if uncommitted)."""
        name = self._staged.get(perturbation, self.metadata["hash_files"][perturbation])
        if name is None:
            raise KeyError(f"Perturbation {perturbation} has no table yet.")
        return self._load_segment(name)

    def _delta_names(self, perturbation: int) -> list[str]:
        if perturbation in self._staged_deltas:
            return self._staged_deltas[perturbation]
        return list(self.metadata["delta_files"][perturbation])

    def load_deltas(self, perturbation: int) -> list[SortedSegment]:
        """The delta segments of ``perturbation`` (oldest first), memory-mapped (staged if uncommitted)."""
        return [self._load_segment(name) for name in self._delta_names(perturbation)]

    def load_counts(self) -> np.ndarray:
        name = self.metadata["counts_file"]
        if name is None:
            return np.zeros(self.n_vectors, dtype=np.uint32)
        if name.endswith(".npy"):  # written before counts were appendable
            return np.load(self.path / name)
        counts = np.fromfile(
            self.path / name, dtype=_COUNTS_DTYPE, count=self.n_vectors
        )
        if len(counts) != self.n_vectors:
            raise ValueError(f"{name} holds fewer than {self.n_vectors} counts.")
        return counts.astype(np.uint32, copy=False)

    # --- writing ---------------------------------------------------------

    def _next_name(self, stem: str, suffix: str = ".npy") -> str:
        self._generation += 1
        return f"{stem}.g{self._generation}{suffix}"

    def _write_counts(self, counts: np.ndarray, unchanged_rows: int) -> str:
        """Write ``counts`` for a commit; returns the counts file name.

        Appends to the current file if the counts of every committed vector are
        unchanged (``unchanged_rows >= n_vectors``), else writes a new file.
        Bytes past the committed counts (left by an interrupted commit) are cut
        off first; the committed ones are never overwritten.
        """
        counts = np.asarray(counts).astype(_COUNTS_DTYPE, copy=False)
        committed = self.n_vectors
        name = self.metadata["counts_file"]
        if (
            name is not None
            and name.endswith(".bin")
            and unchanged_rows >= committed
            and len(counts) >= committed
        ):
            with open(self.path / name, "r+b") as f:
                f.truncate(committed * _COUNTS_DTYPE.itemsize)
                f.seek(0, os.SEEK_END)
                f.write(counts[committed:].tobytes())
            return name
        name = self._next_name("counts", ".bin")
        with open(self.path / name, "wb") as f:
            f.write(counts.tobytes())
        return name

    def _save(self, name: str, array: np.ndarray) -> None:
        with open(self.path / name, "wb") as f:
            np.save(f, array)

    def _fsync_files(self, names) -> None:
        """Make written files durable before a commit refers to them."""
        if not FSYNC:
            return

        def fsync(name):
            fd = os.open(self.path / name, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

        names = list(names)
        with ThreadPoolExecutor(max(1, min(FSYNC_THREADS, len(names)))) as pool:
            list(pool.map(fsync, names))

    def _save_segment(self, name: str, segment: np.ndarray) -> None:
        """Write a hash segment and its fence (computed from the in-memory array)."""
        segment = np.asarray(segment, dtype=np.uint64)
        self._save(name, segment)
        data_offset = (self.path / name).stat().st_size - segment.nbytes
        self._save(fence_name(name), SortedSegment.fence_for(segment, data_offset))

    def write_table(self, perturbation: int, table: np.ndarray) -> None:
        """Stage a new base table for ``perturbation`` (holding all its hashes, so
        it replaces its delta segments too); it takes effect at the next :meth:`commit`."""
        name = self._next_name(f"p{perturbation:04d}")
        self._save_segment(name, table)
        self._staged[perturbation] = name
        self._staged_deltas[perturbation] = []

    def write_delta(
        self, perturbation: int, segment: np.ndarray, replace_last: int = 0
    ) -> None:
        """Stage ``segment`` as the newest delta of ``perturbation``, in place of its
        last ``replace_last`` deltas (whose hashes it must then include)."""
        names = self._delta_names(perturbation)
        if not 0 <= replace_last <= len(names):
            raise ValueError(
                f"Cannot replace {replace_last} of {len(names)} delta segments."
            )
        name = self._next_name(f"p{perturbation:04d}")
        self._save_segment(name, segment)
        self._staged_deltas[perturbation] = [*names[: len(names) - replace_last], name]

    def commit(
        self, counts: np.ndarray, store: VectorStore, unchanged_rows: int = 0
    ) -> None:
        """Make the staged tables and ``counts`` (one per indexed vector) current, atomically.

        ``unchanged_rows``: leading vectors whose counts are the committed ones;
        if that covers every committed vector, only the new counts are written.
        """
        n_vectors = len(counts)
        counts_name = self._write_counts(counts, unchanged_rows)
        new_deltas = {
            name
            for perturbation, names in self._staged_deltas.items()
            for name in names
            if name not in self.metadata["delta_files"][perturbation]
        }
        segments = [*self._staged.values(), *new_deltas]
        self._fsync_files(
            [counts_name, *segments, *(fence_name(name) for name in segments)]
        )

        metadata = dict(self.metadata)
        hash_files = list(metadata["hash_files"])
        for perturbation, name in self._staged.items():
            hash_files[perturbation] = name
        delta_files = [list(names) for names in metadata["delta_files"]]
        for perturbation, names in self._staged_deltas.items():
            delta_files[perturbation] = list(names)
        metadata.update(
            format_version=FORMAT_VERSION,
            hash_files=hash_files,
            delta_files=delta_files,
            counts_file=counts_name,
            generation=self._generation,
            updated=_now(),
        )
        if n_vectors != self.n_vectors:
            metadata["n_vectors"] = n_vectors
            metadata["fingerprint"] = dataset_fingerprint(
                store, n_vectors, previous=self.metadata["fingerprint"]
            )
        self.metadata = metadata
        self._write_metadata()
        self._staged = {}
        self._staged_deltas = {}
        self._remove_unreferenced_files()

    def _write_metadata(self) -> None:
        tmp = self.path / f"{METADATA_FILE}.tmp"
        with open(tmp, "w") as f:
            json.dump(self.metadata, f, indent=1)
            f.flush()
            if FSYNC:
                os.fsync(f.fileno())
        os.replace(tmp, self.path / METADATA_FILE)
        _fsync_dir(self.path)

    def _remove_unreferenced_files(self) -> None:
        keep = set(self.metadata["hash_files"]) | {self.metadata["counts_file"]}
        keep |= {name for names in self.metadata["delta_files"] for name in names}
        keep |= set(self._staged.values())
        keep |= {name for names in self._staged_deltas.values() for name in names}
        keep |= {fence_name(name) for name in keep if name}
        for f in self.path.iterdir():
            if (_DATA_FILE.match(f.name) and f.name not in keep) or _TMP_FILE.match(
                f.name
            ):
                # an open memory map keeps the data alive on POSIX; elsewhere the
                # file is left for the next cleanup
                with contextlib.suppress(OSError):
                    f.unlink()
