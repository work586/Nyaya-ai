"""Usage:  python merge_nyaya.py index.html
Creates index_new.html = your index.html + nyaya-dashboard-patch.html,
inserted right before </body>. Your original file is not modified."""
import sys, pathlib
src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "index.html")
patch = pathlib.Path(__file__).with_name("nyaya-dashboard-patch.html").read_text(encoding="utf-8")
html = src.read_text(encoding="utf-8")
if "book-scene" in html:
    sys.exit("Patch already applied to this file.")
i = html.rfind("</body>")
if i == -1:
    sys.exit("No </body> tag found.")
out = src.with_name("index_new.html")
out.write_text(html[:i] + patch + "\n" + html[i:], encoding="utf-8")
print("Written:", out)