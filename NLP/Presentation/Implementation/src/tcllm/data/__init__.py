"""Data loading, canonical schema and the official-split leakage guards."""

from .loaders import load_processed_bundle, load_split_bundle, save_split_bundle
from .splits import (
    SPLIT_NAMES,
    Example,
    LeakageError,
    SplitBundle,
    assert_no_test_leakage,
    assert_official_sizes,
    normalize_for_dedup,
    overlap_report,
)

__all__ = [
    "SPLIT_NAMES",
    "Example",
    "LeakageError",
    "SplitBundle",
    "assert_no_test_leakage",
    "assert_official_sizes",
    "load_processed_bundle",
    "load_split_bundle",
    "normalize_for_dedup",
    "overlap_report",
    "save_split_bundle",
]
