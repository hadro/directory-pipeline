"""Which collections under ``output/`` count as evaluation data.

``output/`` holds production volumes alongside source probes, five-page evidence
slices, backup directories and one-off genre experiments. Averaging those
together describes nothing: a cross-collection table once reported a 6,862-file
backup directory as a collection, and a rule calibrated on manuscript tests would
have been wrong for every printed directory in the corpus.

The classification lives in ``analysis/collection_registry.txt`` — in the repo
rather than beside the data, because ``output/`` is gitignored and a judgement
about what counts as evidence should be reviewable in a diff.

Anything matching no rule is included. Production is the norm, and a new volume
must not vanish from an evaluation because nobody registered it; the tools print
what they excluded so the opposite mistake stays visible.

    from utils.collections import eval_collections
    kept, excluded = eval_collections(Path("output"))
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "analysis" / "collection_registry.txt"


@dataclass(frozen=True)
class Rule:
    verb: str        # "include" or "exclude"
    glob: str
    category: str


def load_rules(path: "Path | None" = None) -> "list[Rule]":
    """Parse the registry. A missing file means no rules — everything is kept."""
    path = path or REGISTRY_PATH
    rules: list[Rule] = []
    if not path.exists():
        return rules
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2 or parts[0] not in ("include", "exclude"):
            continue
        verb, glob = parts[0], parts[1]
        category = parts[2] if len(parts) > 2 else verb
        rules.append(Rule(verb, glob, category))
    return rules


def classify(name: str, rules: "list[Rule] | None" = None) -> "tuple[bool, str]":
    """Return ``(keep, category)`` for a collection directory name.

    ``include`` rules are checked first, so a specific include can carve one
    volume out of a broad exclude.
    """
    rules = load_rules() if rules is None else rules
    for rule in rules:
        if rule.verb == "include" and fnmatch.fnmatch(name, rule.glob):
            return True, rule.category
    for rule in rules:
        if rule.verb == "exclude" and fnmatch.fnmatch(name, rule.glob):
            return False, rule.category
    return True, "production"


def eval_collections(
    root: Path, rules: "list[Rule] | None" = None
) -> "tuple[list[Path], list[tuple[str, str]]]":
    """Split the immediate subdirectories of *root* into kept and excluded.

    Returns ``(kept_paths, [(name, category), ...])`` — the second list is what
    to report, so an exclusion is never silent.
    """
    rules = load_rules() if rules is None else rules
    kept: list[Path] = []
    dropped: list[tuple[str, str]] = []
    if not root.is_dir():
        return kept, dropped
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        keep, category = classify(d.name, rules)
        (kept.append(d) if keep else dropped.append((d.name, category)))
    return kept, dropped


def summarize_exclusions(dropped: "list[tuple[str, str]]") -> str:
    """One line per category, for a tool's header. Empty string when nothing was dropped."""
    if not dropped:
        return ""
    counts: dict[str, int] = {}
    for _, category in dropped:
        counts[category] = counts.get(category, 0) + 1
    parts = ", ".join(f"{n} {cat}" for cat, n in sorted(counts.items()))
    return (f"Excluded {len(dropped)} collection(s) per analysis/collection_registry.txt "
            f"({parts}).")
