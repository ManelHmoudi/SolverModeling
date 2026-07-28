"""Shared deterministic seeds for benchmark validation runs.

Every algorithm's runner (NSGA-III, QINSGA3, ...) must import SEEDS from
here rather than defining its own list, so that "run N" always means the
exact same seed across algorithms — required for a fair run-by-run
comparison.

First 20 match Cui et al. (2025)'s protocol; 10 more appended to extend the
run count beyond the article without invalidating the original 20 runs.
"""

SEEDS = [
    42, 137, 271, 491, 613, 733, 857, 977, 1009, 1123,
    1249, 1373, 1499, 1609, 1733, 1871, 1997, 2113, 2237, 2351,
    2467, 2593, 2711, 2837, 2953, 3079, 3191, 3313, 3433, 3557,
]
