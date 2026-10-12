try:
    # Written by the build backend (hatch-vcs) at build/install time;
    # missing only in a source tree that was never installed.
    from deduplicate_lib.version import __version__
except ImportError:  # pragma: no cover
    __version__ = "unknown"
