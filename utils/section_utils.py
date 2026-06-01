"""Section boundary utilities for multi-section city directory volumes.

A sections.txt file marks where each structural section begins within a volume:
    # comment lines are ignored
    0015_p15020coll12:2453.jpg  alphabetical
    0181_p15020coll12:2619.jpg  street
    0311_p15020coll12:2749.jpg  business

Each non-comment line contains a filename (first page of that section) and a
label separated by whitespace.  The label is used as a suffix for per-section
prompt files:  ocr_prompt_{label}.md  and  ner_prompt_{label}.md.

Volumes without a sections.txt behave exactly as before.
"""

from __future__ import annotations

from pathlib import Path


def load_sections(sections_path: Path, all_files: list[str]) -> list[dict]:
    """Parse sections.txt and return a list of section dicts.

    Each dict has:
        label        str    — section name (e.g. "alphabetical")
        first_file   str    — filename of the first page in the section
        start_idx    int    — index into all_files for first_file
        end_idx      int    — index of last page in section (inclusive)
        page_indices list[int]  — sorted list of indices covered by this section
    """
    lines: list[tuple[str, str]] = []
    with open(sections_path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                lines.append((parts[0], parts[1].strip()))

    if not lines:
        return []

    file_index: dict[str, int] = {f: i for i, f in enumerate(all_files)}

    sections: list[dict] = []
    for filename, label in lines:
        idx = file_index.get(filename)
        if idx is None:
            # Try matching by basename only in case paths differ
            for i, f in enumerate(all_files):
                if Path(f).name == Path(filename).name:
                    idx = i
                    break
        if idx is None:
            raise ValueError(
                f"sections.txt references '{filename}' which was not found in the image list"
            )
        sections.append({"label": label, "first_file": filename, "start_idx": idx})

    sections.sort(key=lambda s: s["start_idx"])

    for i, sec in enumerate(sections):
        if i + 1 < len(sections):
            sec["end_idx"] = sections[i + 1]["start_idx"] - 1
        else:
            sec["end_idx"] = len(all_files) - 1
        sec["page_indices"] = list(range(sec["start_idx"], sec["end_idx"] + 1))

    return sections


def section_for_page(filename: str, sections: list[dict], all_files: list[str]) -> str | None:
    """Return the section label for the given page filename, or None if no sections."""
    if not sections:
        return None
    file_index: dict[str, int] = {f: i for i, f in enumerate(all_files)}
    idx = file_index.get(filename)
    if idx is None:
        # fallback: match by basename
        for i, f in enumerate(all_files):
            if Path(f).name == Path(filename).name:
                idx = i
                break
    if idx is None:
        return None
    for sec in reversed(sections):
        if idx >= sec["start_idx"]:
            return sec["label"]
    return None


def prompt_for_page(
    filename: str,
    sections: list[dict],
    all_files: list[str],
    slug_dir: Path,
    prompt_type: str,
) -> Path:
    """Return the prompt path for this page.

    Looks up the section label, then returns:
        {slug_dir}/{prompt_type}_{label}.md   if it exists
    Fallback chain:
        {slug_dir}/{prompt_type}.md
        None  (caller should use its own default)

    prompt_type is 'ocr_prompt' or 'ner_prompt'.
    """
    label = section_for_page(filename, sections, all_files)
    if label:
        specific = slug_dir / f"{prompt_type}_{label}.md"
        if specific.exists():
            return specific
    generic = slug_dir / f"{prompt_type}.md"
    if generic.exists():
        return generic
    return slug_dir / f"{prompt_type}.md"


def is_section_boundary(
    filename: str,
    sections: list[dict],
    all_files: list[str],
) -> bool:
    """Return True if this page is the first page of a section (other than the first)."""
    if not sections:
        return False
    file_index: dict[str, int] = {f: i for i, f in enumerate(all_files)}
    idx = file_index.get(filename)
    if idx is None:
        for i, f in enumerate(all_files):
            if Path(f).name == Path(filename).name:
                idx = i
                break
    if idx is None:
        return False
    return any(sec["start_idx"] == idx for sec in sections[1:])
