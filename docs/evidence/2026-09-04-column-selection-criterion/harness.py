"""Load a page's Surya + Gemini inputs and mirror align_image's call into
build_aligned_lines, so both candidate alignments can be scored side by side.

Imported by compare_criteria.py; the names below are its public surface.
"""
import sys,os,json,glob,re

__all__ = ["RAW", "FIXED", "FRAG", "load", "prep",
           "plan_columns", "_legacy_reading_order",
           "inversion_rate", "bbox_centre"]
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline.align_ocr import build_aligned_lines as FIXED, plan_columns, _legacy_reading_order
# The raw column-only candidate cannot come from the current build_aligned_lines:
# the selection lives inside it, so calling it measures the choice, not the
# candidates. Materialise the pre-fix module from the commit that preceded this
# work. Pinned rather than HEAD, because HEAD moves once the fix is committed.
PREFIX_COMMIT = "c77bd94"
_prefix = os.path.join(os.path.dirname(os.path.abspath(__file__)), "align_prefix.py")
if not os.path.exists(_prefix):
    import subprocess
    _repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    with open(_prefix, "w") as fh:
        subprocess.run(["git", "show", f"{PREFIX_COMMIT}:pipeline/align_ocr.py"],
                       cwd=_repo, stdout=fh, check=True)
import align_prefix
RAW=align_prefix.build_aligned_lines
from utils.column_utils import inversion_rate, bbox_centre
FRAG=lambda b: f"#xywh={b[0]},{b[1]},{b[2]-b[0]},{b[3]-b[1]}"

def load(p):
    stem=p.replace('_aligned.json','')
    if not os.path.exists(stem+'.txt'): return None
    sp=stem.rsplit('_gemini',1)[0]; g=glob.glob(sp+'*_surya.json')
    if not g: return None
    s=json.load(open(g[0]))
    sl=[l for l in (s.get('lines') or s.get('text_lines') or []) if 'bbox' in l and l.get('text')]
    W=s.get('image_width') or 0
    if not sl or not W: return None
    gl=[];ad=set();inad=False
    for raw in open(stem+'.txt',encoding='utf-8',errors='replace').read().splitlines():
        ln=raw.strip()
        if not ln: continue
        if re.match(r"^=== .+ ===$",ln): inad=ln.startswith("=== ADVERTISEMENT"); continue
        if inad: ad.add(len(gl))
        gl.append(ln)
    return sl,W,gl,ad

def prep(sl,W):
    """Mirror align_image exactly."""
    plan=plan_columns(sl,W)
    if plan is None: return None
    header,cols=plan
    groups=[header]+cols
    fb=_legacy_reading_order(sl,W)
    lines=header+[l for c in cols for l in c]
    return lines,groups,fb
