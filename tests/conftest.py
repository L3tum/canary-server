"""Pytest collection configuration.

Some files in tests/ are executable integration/load-test scripts that require a
running server or heavyweight optional dependencies. Keep the default pytest
suite focused on isolated unit/API tests.
"""

collect_ignore = [
    "test_cpu_optimization.py",
    "test_endpoint.py",
    "test_parallel_loading.py",
]
