"""pipeline/align_ocr.py — column planning and column-partitioned alignment."""

from pipeline.align_ocr import (
    build_aligned_lines,
    plan_columns,
    sort_by_reading_order,
)

W = 2048


def _page(n_cols, rows=30, starts=(130, 760, 1380), width=430, header=None):
    """Synthetic multi-column page; text is unique per line so NW can anchor."""
    lines = []
    if header:
        for i, (text, bbox) in enumerate(header):
            lines.append({"bbox": list(bbox), "text": text})
    for c in range(n_cols):
        x = starts[c]
        for r in range(rows):
            y = 200 + r * 40
            lines.append({
                "bbox": [x, y, x + width, y + 30],
                "text": f"col{c}row{r} entry",
            })
    return lines


def _reading_texts(lines):
    return [ln["text"] for ln in sort_by_reading_order(lines, W)]


# ── plan_columns ──────────────────────────────────────────────────────────

def test_detects_three_columns():
    plan = plan_columns(_page(3), W)
    assert plan is not None
    header, cols = plan
    assert header == []
    assert len(cols) == 3


def test_three_columns_emitted_column_major():
    texts = _reading_texts(_page(3, rows=5))
    assert texts[:5] == [f"col0row{r} entry" for r in range(5)]
    assert texts[5:10] == [f"col1row{r} entry" for r in range(5)]


def test_sparse_bridge_lines_do_not_hide_a_gutter():
    """A consecutive-x1-gap rule misses gutters bridged by stray indents."""
    lines = _page(3, rows=20)
    for x in (600, 900, 1100, 1250):          # sparse lines inside the gutters
        lines.append({"bbox": [x, 1500, x + 120, 1530], "text": f"stray{x}"})
    plan = plan_columns(lines, W)
    assert plan is not None and len(plan[1]) == 3


def test_single_column_page_returns_none():
    lines = [{"bbox": [130, 200 + i * 40, 1900, 230 + i * 40], "text": f"line{i}"}
             for i in range(30)]
    assert plan_columns(lines, W) is None


def test_wide_centred_header_is_hoisted_ahead_of_columns():
    header = [("STATE HEADING", (500, 60, 1600, 130))]
    texts = _reading_texts(_page(3, rows=5, header=header))
    assert texts[0] == "STATE HEADING"


def test_narrow_gutter_crossing_line_is_not_hoisted():
    """Wrapped city-directory entries cross a gutter but are not headers."""
    lines = _page(3, rows=20)
    lines.append({"bbox": [500, 900, 900, 930], "text": "wrapped entry tail"})
    header, _ = plan_columns(lines, W)
    assert [h["text"] for h in header] == []


def test_narrow_running_head_above_the_columns_is_hoisted():
    """A running head ("OHIO-Continued") is narrower than the wide-header rule
    but still belongs ahead of the columns, not ~50 lines into one.

    Geometry mirrors the real case: the head sits in the middle column and
    crosses the *second* gutter, so it lands in a column that counts toward
    body_top.
    """
    head = [("OHIO-Continued", (760, 60, 1330, 100))]   # 27.8% of page width
    texts = _reading_texts(_page(3, rows=8, header=head))
    assert texts[0] == "OHIO-Continued"


def test_running_head_does_not_set_its_own_body_top():
    """Regression: the head is gutter-spanning, so measuring body_top over all
    lines let it define the very threshold that would exempt it, and it could
    never qualify.  A page number level with it must not resurrect that."""
    head = [("TEXAS-Continued", (760, 60, 1330, 100))]
    lines = _page(3, rows=8, header=head)
    lines.append({"bbox": [130, 64, 190, 96], "text": "450"})   # page number
    header, _ = plan_columns(lines, W)
    assert "TEXAS-Continued" in [h["text"] for h in header]


def test_mid_page_gutter_crossing_line_is_not_hoisted():
    """The top-band constraint keeps stray mid-page fragments out."""
    lines = _page(3, rows=20)
    # same width and gutter-crossing as a running head, but halfway down
    lines.append({"bbox": [760, 640, 1330, 680], "text": "AMITY ST."})
    header, _ = plan_columns(lines, W)
    assert "AMITY ST." not in [h["text"] for h in header]


def test_narrow_head_declined_when_hoist_already_over_collects():
    """Green-book listing pages already hoist body lines past the partition
    guard; adding more only makes the plan worse, so leave them unchanged."""
    lines = _page(3, rows=8)
    # wide gutter-spanning body lines the strict rule already hoists, enough to
    # push the header group past _PARTITION_MAX_HEADER_FRAC
    for i in range(4):
        y = 400 + i * 40
        lines.append({"bbox": [500, y, 1400, y + 30], "text": f"listing {i}"})
    lines.append({"bbox": [760, 60, 1330, 100], "text": "NARROW HEAD"})
    plan = plan_columns(lines, W)
    assert plan is not None
    header, _ = plan
    assert "NARROW HEAD" not in [h["text"] for h in header]


# ── build_aligned_lines: guards ───────────────────────────────────────────

def _frag(bbox):
    return ""


def _groups(lines_plan):
    header, cols = lines_plan
    return [header] + cols


def test_partitioned_alignment_places_each_column_independently():
    lines = _page(3, rows=6)
    plan = plan_columns(lines, W)
    gem = [f"col{c}row{r} entry" for c in range(3) for r in range(6)]
    out, unmatched = build_aligned_lines(
        sort_by_reading_order(lines, W), gem, _frag,
        column_groups=_groups(plan), fallback_order=lines,
    )
    by_bbox = {tuple(o["bbox"]): o["gemini_text"] for o in out}
    for ln in lines:
        assert by_bbox[tuple(ln["bbox"])] == ln["text"]
    assert unmatched == []


def _unpartitioned(lines, gem):
    return build_aligned_lines(lines, gem, _frag)


def test_merge_ratio_guard_reverts_to_whole_page_alignment():
    """Gemini merging rows across columns must revert to the fallback order."""
    lines = _page(3, rows=20)                         # 60 Surya lines
    plan = plan_columns(lines, W)
    gem = [f"col0row{r} entry" for r in range(20)]    # ratio 3.0 >= 1.5
    fallback = sort_by_reading_order(lines, W)
    out, um = build_aligned_lines(
        sort_by_reading_order(lines, W), gem, _frag,
        column_groups=_groups(plan), fallback_order=fallback,
    )
    expected, expected_um = _unpartitioned(fallback, gem)
    assert [o["gemini_text"] for o in out] == [e["gemini_text"] for e in expected]
    assert [o["bbox"] for o in out] == [e["bbox"] for e in expected]
    assert um == expected_um


def test_header_fraction_guard_reverts_when_hoist_over_collects():
    """A header group larger than 10% of the page disables partitioning."""
    lines = _page(2, rows=20, starts=(130, 760))      # 40 lines
    plan = plan_columns(lines, W)
    _, cols = plan
    # Pathological plan: a third of the page hoisted as "header".
    groups = [cols[0][:14], cols[0][14:], cols[1]]
    assert len(groups[0]) / len(lines) > 0.10
    gem = [ln["text"] for ln in sort_by_reading_order(lines, W)]
    fallback = sort_by_reading_order(lines, W)
    out, _ = build_aligned_lines(
        list(fallback), gem, _frag,
        column_groups=groups, fallback_order=fallback,
    )
    expected, _ = _unpartitioned(fallback, gem)
    assert [o["bbox"] for o in out] == [e["bbox"] for e in expected]


def test_unlocatable_column_reverts_rather_than_guessing():
    """A column whose text is absent from the Gemini side must not be forced."""
    lines = _page(3, rows=8)
    plan = plan_columns(lines, W)
    # Gemini saw columns 0 and 1 only; column 2 has no counterpart text.
    gem = [f"col{c}row{r} entry" for c in (0, 1) for r in range(8)]
    fallback = sort_by_reading_order(lines, W)
    out, _ = build_aligned_lines(
        list(fallback), gem, _frag,
        column_groups=_groups(plan), fallback_order=fallback,
    )
    placed = {tuple(o["bbox"]): o["gemini_text"] for o in out}
    # Whatever it does, it must not put column-2 bboxes on column-0/1 text.
    for ln in lines:
        if ln["text"].startswith("col2"):
            assert placed.get(tuple(ln["bbox"]), "col2") .startswith("col2")


def test_no_column_groups_matches_unpartitioned_path():
    lines = _page(2, rows=8, starts=(130, 760))
    gem = [ln["text"] for ln in sort_by_reading_order(lines, W)]
    a, _ = build_aligned_lines(sort_by_reading_order(lines, W), gem, _frag)
    b, _ = build_aligned_lines(sort_by_reading_order(lines, W), gem, _frag,
                               column_groups=None, fallback_order=None)
    assert [x["gemini_text"] for x in a] == [x["gemini_text"] for x in b]
