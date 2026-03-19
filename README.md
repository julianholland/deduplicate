<div align="center">
  <h1><code>deduplicate</code></h1>
  <p><i>deduplication algorithms in python</i></p>
</div>

***
## Key Features

- Factory Plugin architecture, for easy extensibility and modification
- Suite of tolerance tuning algorithms to help you find the right tolerance value for your system (not yet implemented)
- Suite of benchmarking tools to ensure rigor, accuracy, and speed (not yet implemented)


***
## Implemented Algorithms

- Distance Matrix (Simple, accurate, expensive): Computes the distance matrix for all vectors and determines duplicates by finding those that fall below a given distance
- Multi Hashing (Fast): Smears and rounds the vectors using a normal distribution and computes the hashes for each which are then used to determine duplicates by proportion of hash clashes. (not yet implemented)
- Locality Sensitive Hashing (Fast, Accurate)
