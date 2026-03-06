#!/usr/bin/env bash
# make-git-repo.sh — Assemble pipeline output into a GitHub Pages deployable folder.
#
# Usage:
#   ./make-git-repo.sh <ITEM_DIR> <DEST_DIR> <GITHUB_PAGES_URL>
#
# Arguments:
#   ITEM_DIR          Path to the item subdirectory that contains manifest.json
#                     (e.g. output/hackley_.../4f7822b0-...)
#   DEST_DIR          Destination folder to populate (created if absent)
#   GITHUB_PAGES_URL  Base URL where the repo will be published
#                     (e.g. https://hadro.github.io/my-repo)
#
# The script copies/generates:
#   manifest.json
#   *_box_annotations.json
#   ranges_*.json           (if present)
#   entries_*_geocoded.html → map.html  (with <title> and <h1> updated)
#   index.html              (Mirador 3 viewer with IIIF Content State support)

set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────────────────
if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <ITEM_DIR> <DEST_DIR> <GITHUB_PAGES_URL>" >&2
  exit 1
fi

ITEM_DIR="${1%/}"        # strip trailing slash
DEST_DIR="${2%/}"
GITHUB_PAGES_URL="${3%/}"

if [[ ! -d "$ITEM_DIR" ]]; then
  echo "Error: ITEM_DIR '$ITEM_DIR' not found." >&2
  exit 1
fi

MANIFEST="$ITEM_DIR/manifest.json"
if [[ ! -f "$MANIFEST" ]]; then
  echo "Error: manifest.json not found in '$ITEM_DIR'." >&2
  exit 1
fi

mkdir -p "$DEST_DIR"

# ── Extract title and year from manifest.json ─────────────────────────────────
META=$(python3 - "$MANIFEST" <<'PYEOF'
import sys, json, re

manifest_path = sys.argv[1]
with open(manifest_path) as f:
    m = json.load(f)

# Title — IIIF 3: label is {lang: [str, ...]}
label = m.get("label", {})
if isinstance(label, dict):
    candidates = next(iter(label.values()), [])
    title = candidates[0] if candidates else ""
elif isinstance(label, str):
    title = label
else:
    title = ""
title = re.sub(r"<[^>]+>", "", title).strip()

# Year — look in metadata for a "Dates / Origin" entry with "Date Issued: YYYY"
year = ""
for entry in m.get("metadata", []):
    lbl_vals = entry.get("label", {})
    lbl_text = " ".join(
        v for vals in (lbl_vals.values() if isinstance(lbl_vals, dict) else [[str(lbl_vals)]])
        for v in vals
    )
    if "date" in lbl_text.lower() or "origin" in lbl_text.lower():
        vals = entry.get("value", {})
        val_text = " ".join(
            v for vs in (vals.values() if isinstance(vals, dict) else [[str(vals)]])
            for v in vs
        )
        val_text = re.sub(r"<[^>]+>", "", val_text)
        m2 = re.search(r"Date Issued[:\s]+(\d{4})", val_text, re.IGNORECASE)
        if m2:
            year = m2.group(1)
            break

print(title)
print(year)
PYEOF
)

ITEM_TITLE=$(printf '%s\n' "$META" | head -1)
ITEM_YEAR=$(printf '%s\n' "$META" | tail -1)

# Fallback: use directory name if title is empty or just an ID
if [[ -z "$ITEM_TITLE" || "$ITEM_TITLE" =~ ^[0-9a-f-]+$ ]]; then
  ITEM_TITLE="$(basename "$(dirname "$ITEM_DIR")")"
  # Convert underscores to spaces and title-case
  ITEM_TITLE="$(echo "$ITEM_TITLE" | sed 's/_/ /g' | sed 's/\b\(.\)/\u\1/g')"
fi

DISPLAY_TITLE="$ITEM_TITLE"
[[ -n "$ITEM_YEAR" ]] && DISPLAY_TITLE="$ITEM_TITLE ($ITEM_YEAR)"

echo "Title : $DISPLAY_TITLE"
echo "Dest  : $DEST_DIR"
echo "URL   : $GITHUB_PAGES_URL"
echo ""

# ── Copy static files ─────────────────────────────────────────────────────────
# Copy manifest and patch its id to match the GitHub Pages URL.
# The manifest id must equal the URL where it is served; if it points to the
# source institution (e.g. loc.gov) Mirador will try to re-fetch from there
# and hit CORS/403 errors, causing "An error occurred".
python3 - "$MANIFEST" "$MANIFEST_URL" <<'PYEOF' > "$DEST_DIR/manifest.json"
import json, sys
with open(sys.argv[1]) as f:
    m = json.load(f)
# Patch manifest id to the GitHub Pages URL (IIIF spec requires id == served URL;
# if it points to the source institution Mirador will try to re-fetch and hit 403).
m["id"] = sys.argv[2]
# Fix LoC manifests that incorrectly declare ImageService3 for tiles that actually
# serve IIIF Image API 2.  Mirador's ThumbnailFactory crashes on the mismatch.
fixed = 0
for canvas in m.get("items", []):
    for ann_page in canvas.get("items", []):
        for ann in ann_page.get("items", []):
            body = ann.get("body", {})
            for svc in body.get("service", []):
                if svc.get("type") == "ImageService3" and "maxWidth" in svc:
                    svc["type"] = "ImageService2"
                    svc.pop("maxWidth", None)
                    svc["profile"] = "http://iiif.io/api/image/2/level2.json"
                    fixed += 1
if fixed:
    import sys as _sys
    print(f"  (patched {fixed} ImageService3→ImageService2 declarations)", file=_sys.stderr)
# Add sequential labels to canvases that lack them so Mirador shows page numbers
# instead of "NaN" in the thumbnail strip.
import re as _re
labeled = 0
for i, canvas in enumerate(m.get("items", [])):
    if not canvas.get("label"):
        m2 = _re.search(r"/canvas/(\d+)$", canvas.get("id", ""))
        canvas["label"] = {"en": [m2.group(1) if m2 else str(i + 1)]}
        labeled += 1
if labeled:
    import sys as _sys2
    print(f"  (added labels to {labeled} unlabeled canvases)", file=_sys2.stderr)
print(json.dumps(m, ensure_ascii=False, indent=2))
PYEOF
echo "  Copied manifest.json (id → ${MANIFEST_URL})"

shopt -s nullglob

# Box annotation files
ANNOTATIONS=("$ITEM_DIR"/*_box_annotations.json)
if [[ ${#ANNOTATIONS[@]} -gt 0 ]]; then
  cp "${ANNOTATIONS[@]}" "$DEST_DIR/"
  echo "  Copied ${#ANNOTATIONS[@]} annotation file(s)"
fi

# Ranges files
RANGES=("$ITEM_DIR"/ranges_*.json)
if [[ ${#RANGES[@]} -gt 0 ]]; then
  cp "${RANGES[@]}" "$DEST_DIR/"
  echo "  Copied ${#RANGES[@]} ranges file(s)"
fi

# ── Copy and patch map.html ───────────────────────────────────────────────────
GEOCODED=("$ITEM_DIR"/entries_*_geocoded.html)
if [[ ${#GEOCODED[@]} -gt 0 ]]; then
  # Use the last file if there are multiple (bash 3 compatible)
  LAST_IDX=$((${#GEOCODED[@]} - 1))
  MAP_SRC="${GEOCODED[$LAST_IDX]}"
  # Escape special sed characters in the replacement string
  ESCAPED_TITLE="$(printf '%s\n' "$DISPLAY_TITLE" | sed 's/[&/\\]/\\&/g')"
  sed \
    -e "s|<title>Green Book Map</title>|<title>${ESCAPED_TITLE}</title>|" \
    -e "s|<h1>Negro Motorist Green Book</h1>|<h1>${ESCAPED_TITLE}</h1>|" \
    -e "s|const YEAR_LABEL\( *\)= \"[^\"]*\";|const YEAR_LABEL\1= \"${ESCAPED_TITLE}\";|" \
    "$MAP_SRC" > "$DEST_DIR/map.html"
  echo "  Wrote map.html (from $(basename "$MAP_SRC"))"
else
  echo "  Warning: no entries_*_geocoded.html found — map.html not written"
fi

# ── Generate index.html (Mirador viewer) ─────────────────────────────────────
MANIFEST_URL="${GITHUB_PAGES_URL}/manifest.json"

# Use a quoted heredoc so bash variables are substituted before the JS content
cat > "$DEST_DIR/index.html" << ENDOFHTML
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>${DISPLAY_TITLE} — IIIF Viewer</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    html, body, #mirador { height: 100%; }
  </style>
</head>
<body>
  <div id="mirador"></div>

  <script src="https://unpkg.com/mirador@3.3.0/dist/mirador.min.js"></script>
  <script>
    // Decode optional IIIF Content State parameter (?iiif-content=BASE64URL).
    // Mirador does not process this automatically — we decode it here and pass
    // canvasId to the window config so the viewer opens at the right canvas.
    function decodeContentState(encoded) {
      try {
        var b64 = encoded.replace(/-/g, "+").replace(/_/g, "/");
        while (b64.length % 4) b64 += "=";
        return JSON.parse(atob(b64));
      } catch (e) {
        console.warn("iiif-content decode failed:", e);
        return null;
      }
    }

    var DEFAULT_MANIFEST = "${MANIFEST_URL}";

    var windowConfig = {
      manifestId: DEFAULT_MANIFEST,
      thumbnailNavigationPosition: "far-bottom",
      view: "single"
    };

    var iiifContent = new URLSearchParams(window.location.search).get("iiif-content");
    if (iiifContent) {
      var state = decodeContentState(iiifContent);
      if (state && state.target) {
        var target = state.target;
        // Strip any #xywh= fragment before passing to Mirador — Mirador does
        // an exact string match against manifest canvas IDs, which never include
        // fragments, so including the fragment prevents the canvas from being found.
        var rawId = typeof target === "string" ? target : (target.id || "");
        var canvasId = rawId.split("#")[0];
        var manifestId =
          (target.partOf && target.partOf[0] && target.partOf[0].id) ||
          DEFAULT_MANIFEST;
        if (canvasId) windowConfig.canvasId = canvasId;
        windowConfig.manifestId = manifestId;
      }
    }

    Mirador.viewer({
      id: "mirador",
      windows: [windowConfig],
      workspace: { showZoomControls: true },
      workspaceControlPanel: { enabled: false },
      // Tell OpenSeadragon to request tiles with crossOrigin="anonymous" so
      // that WebGL compositing (and annotation overlays) work correctly when
      // tiles are served from a different origin.
      osdConfig: {
        crossOriginPolicy: "Anonymous",
        ajaxWithCredentials: false,
      },
    });
  </script>
</body>
</html>
ENDOFHTML

echo "  Wrote index.html (manifest: ${MANIFEST_URL})"
echo ""
echo "Done. Push '${DEST_DIR}' to GitHub and enable GitHub Pages."
