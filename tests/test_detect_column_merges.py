"""Tests for tools/detect_column_merges.py — the genuine-merge heuristic.

A genuine column merge shows up as one aligned line carrying two directory
entries, most reliably two street addresses. These guard the address-counting
and merge-classification logic against regression.
"""

from tools.detect_column_merges import (
    _address_count,
    _estimate_columns,
    _length_outlier_lines,
    _looks_merged,
    _merged_lines,
)


class TestAddressCount:
    def test_two_entries_two_addresses(self):
        line = "Royal—1073 Fulton St. Carver—980 Prospect Ave."
        assert _address_count(line) >= 2

    def test_single_entry_one_address(self):
        assert _address_count("Carver—980 Prospect Ave.") == 1

    def test_no_address(self):
        assert _address_count("CLUBS AND TAVERNS") == 0

    def test_bare_number_without_street_suffix_not_counted(self):
        # Membership/year figures with no street suffix are not addresses.
        assert _address_count("Organized 1910. Membership 140.") == 0


class TestLooksMerged:
    def test_two_address_line_is_merged(self):
        line = "Parkside 759 Gates Ave. Crotona—695 East 170th St."
        assert _looks_merged(line, median_len=30) is True

    def test_single_entry_not_merged(self):
        assert _looks_merged("Crotona—695 East 170th St.", median_len=30) is False


# ---------------------------------------------------------------------------
# Generic (content-agnostic) signal
# ---------------------------------------------------------------------------

class TestEstimateColumns:
    def test_single_column(self):
        boxes = [[100, y, 400, y + 20] for y in range(0, 240, 20)]  # 12 boxes, x1≈100
        assert _estimate_columns(boxes, page_width=1000) == 1

    def test_two_columns(self):
        left = [[50, y, 450, y + 20] for y in range(0, 200, 20)]    # x1≈50
        right = [[550, y, 950, y + 20] for y in range(0, 200, 20)]  # x1≈550
        assert _estimate_columns(left + right, page_width=1000) == 2

    def test_too_few_boxes_is_one(self):
        assert _estimate_columns([[0, 0, 10, 10]], page_width=1000) == 1


class TestLengthOutlierLines:
    def test_flags_the_long_outlier(self):
        texts = ["alpha beta"] * 8 + ["alpha beta gamma delta epsilon zeta"]
        out = _length_outlier_lines(texts, factor=1.8)
        assert out == ["alpha beta gamma delta epsilon zeta"]

    def test_uniform_lines_yield_nothing(self):
        assert _length_outlier_lines(["alpha beta"] * 10, factor=1.8) == []

    def test_short_page_yields_nothing(self):
        assert _length_outlier_lines(["a b c d e f g"] * 4, factor=1.8) == []


class TestMergedLinesModes:
    # 7 single-address lines + 1 genuine two-address line (also a length outlier).
    TEXTS = ["Cafe—10 A St."] * 7 + ["Alpha House—120 Main St. Beta Inn—340 Oak Ave."]
    MERGED = "Alpha House—120 Main St. Beta Inn—340 Oak Ave."

    def test_address_mode_ignores_column_count(self):
        # Address signal fires regardless of geometry.
        assert _merged_lines(self.TEXTS, "address", 1.8, column_count=1, median_len=13) == [self.MERGED]

    def test_generic_requires_multi_column(self):
        assert _merged_lines(self.TEXTS, "generic", 1.8, column_count=1, median_len=13) == []
        assert _merged_lines(self.TEXTS, "generic", 1.8, column_count=2, median_len=13) == [self.MERGED]

    def test_both_unions_without_double_counting(self):
        # The line is both an address-merge and a length outlier — counted once.
        out = _merged_lines(self.TEXTS, "both", 1.8, column_count=2, median_len=13)
        assert out == [self.MERGED]
