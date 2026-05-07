#!/usr/bin/env bash
# Рендерит report/report.md в report/report.pdf через pandoc + Chrome.
# Требует: pandoc, Google Chrome / Chromium на стандартном macOS-пути.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
MD="$ROOT/report/report.md"
HTML="$ROOT/report/report.html"
PDF="$ROOT/report/report.pdf"

CHROME_PATHS=(
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
  "/Applications/Chromium.app/Contents/MacOS/Chromium"
  "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
)
CHROME=""
for p in "${CHROME_PATHS[@]}"; do
  if [[ -x "$p" ]]; then CHROME="$p"; break; fi
done
if [[ -z "$CHROME" ]]; then
  echo "Chrome/Chromium не найден на стандартных путях." >&2
  exit 1
fi

pandoc "$MD" -o "$HTML" --standalone --metadata title="Credit Default — Project Report"

python3 - "$HTML" <<'PY'
import sys
from pathlib import Path

p = Path(sys.argv[1])
html = p.read_text()
css = """
<style>
  body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1.5em; line-height: 1.55; color: #222; }
  h1, h2, h3 { color: #1a3a6c; }
  h1 { border-bottom: 2px solid #1a3a6c; padding-bottom: 0.2em; }
  h2 { border-bottom: 1px solid #ddd; padding-bottom: 0.1em; margin-top: 1.6em; }
  table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.92em; }
  th, td { border: 1px solid #ccc; padding: 0.4em 0.7em; text-align: left; }
  th { background: #f3f5f8; }
  code { background: #f4f4f4; padding: 0.1em 0.3em; border-radius: 3px; }
  pre { background: #f4f4f4; padding: 0.7em; border-radius: 5px; overflow-x: auto; }
  img { max-width: 100%; height: auto; border: 1px solid #eee; }
  blockquote { border-left: 4px solid #1a3a6c; padding-left: 1em; color: #555; }
  @media print { body { max-width: none; margin: 0; padding: 1cm 1.2cm; } a { color: inherit; text-decoration: none; } img { page-break-inside: avoid; } }
</style>
"""
p.write_text(html.replace("</head>", css + "</head>"))
PY

"$CHROME" --headless --disable-gpu --no-sandbox --no-pdf-header-footer \
  --print-to-pdf="$PDF" --print-to-pdf-no-header \
  "file://$HTML" >/dev/null 2>&1

rm "$HTML"
echo "PDF: $PDF"
