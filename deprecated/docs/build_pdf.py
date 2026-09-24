"""Render architecture_redesign_analysis.md -> styled HTML -> PDF (LibreOffice).

Pure-stdlib + `markdown` (already in the venv). CSS is kept intentionally simple
so LibreOffice's Writer/Web HTML import renders it faithfully (tables, code
blocks, page breaks, shaded emoji-severity cells).

    python docs/build_pdf.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
MD = HERE / "architecture_redesign_analysis.md"
HTML = HERE / "architecture_redesign_analysis.html"
PDF = HERE / "architecture_redesign_analysis.pdf"

# LibreOffice's HTML importer honours a conservative CSS subset: font-family,
# colors, borders, background, margins, page-break-before. Avoid fl<ex/grid.
CSS = """
@page { size: A4; margin: 1.6cm 1.5cm; }
body { font-family: 'Liberation Sans', Arial, sans-serif; font-size: 10.2pt;
       color: #1c2833; line-height: 1.42; }
h1 { font-size: 19pt; color: #0f2c44; border-bottom: 3px solid #0f2c44;
     padding-bottom: 4px; margin-top: 26px; }
h2 { font-size: 14.5pt; color: #103a5c; border-bottom: 1px solid #9db8cc;
     padding-bottom: 2px; margin-top: 22px; }
h3 { font-size: 11.8pt; color: #1a5276; margin-top: 16px; }
h4 { font-size: 10.6pt; color: #21618c; margin-top: 12px; }
p, li { font-size: 10.2pt; }
code { font-family: 'Liberation Mono', 'Courier New', monospace; font-size: 9pt;
       background: #eef2f5; color: #7d2f0d; padding: 0 2px; }
pre { font-family: 'Liberation Mono', 'Courier New', monospace; font-size: 8.4pt;
      background: #f4f6f8; border: 1px solid #c8d3dc; border-left: 4px solid #21618c;
      padding: 8px 10px; color: #14324a; line-height: 1.32; }
pre code { background: transparent; color: #14324a; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; }
th { background: #103a5c; color: #ffffff; font-size: 9pt; text-align: left;
     padding: 5px 7px; border: 1px solid #103a5c; }
td { border: 1px solid #b9c6d1; padding: 4px 7px; font-size: 9pt;
     vertical-align: top; }
tr:nth-child(even) td { background: #f2f6f9; }
hr { border: none; border-top: 1px solid #c8d3dc; margin: 18px 0; }
blockquote { border-left: 4px solid #2e86c1; background: #eef5fb; margin: 10px 0;
             padding: 6px 12px; color: #21384a; font-style: italic; }
strong { color: #0f2c44; }
a { color: #1a5276; }
"""

COVER = """
<div style="text-align:center; padding-top:120px; padding-bottom:60px;">
  <div style="font-size:26pt; color:#0f2c44; font-weight:bold;">
    Compositional Action Models</div>
  <div style="font-size:15pt; color:#21618c; margin-top:10px;">
    Architecture &amp; Redesign Analysis</div>
  <div style="font-size:10.5pt; color:#5d6d7e; margin-top:26px;">
    Component-level map &middot; interface schemas &middot; migration path<br/>
    for the compositional / zero-shot research direction</div>
  <div style="font-size:9pt; color:#85929e; margin-top:60px;">
    Generated 2026-08-15 &nbsp;&middot;&nbsp; branch: multiskill</div>
</div>
<p style="page-break-before: always;"></p>
"""


def main() -> int:
    md_text = MD.read_text()
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "toc", "attr_list"],
    )
    html = (
        "<html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{COVER}{body}</body></html>"
    )
    HTML.write_text(html)
    print(f"wrote {HTML} ({len(html)} bytes)")

    # Force a fresh profile dir so a running LibreOffice instance can't block us.
    profile = HERE / ".lo_profile"
    cmd = [
        "soffice", "--headless", "--norestore",
        f"-env:UserInstallation=file://{profile}",
        "--convert-to", "pdf", "--outdir", str(HERE), str(HTML),
    ]
    print("running:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    sys.stdout.write(res.stdout)
    sys.stderr.write(res.stderr)
    if not PDF.exists():
        print("ERROR: PDF not produced", file=sys.stderr)
        return 1
    print(f"OK -> {PDF} ({PDF.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
