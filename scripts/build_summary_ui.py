#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

REQUIRED_METRICS = [
    "Average_Repeatability",
    "Avg_Uncalibrated_Accuracy",
    "Avg_Calibrated_Accuracy",
]

EXCLUDE_NOTE_PREFIXES = (
    "No specific PN in specs",
    "no base model found in specs",
    "No matching spec key",
)


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def is_note_excluded(note: str) -> bool:
    if not note:
        return False
    note = note.strip()
    return any(note.startswith(pref) for pref in EXCLUDE_NOTE_PREFIXES)


def fully_matched(part_rows: List[Dict[str, str]]) -> bool:
    # Must have all required metrics with concrete spec_max and pass True/False and no exclusion notes
    metrics_present = {m: False for m in REQUIRED_METRICS}
    for r in part_rows:
        m = r.get("metric", "")
        if m not in metrics_present:
            continue
        if is_note_excluded(r.get("note", "")):
            return False
        if not (r.get("spec_max") or "").strip():
            continue
        p = (r.get("pass") or "").strip()
        if p not in ("True", "False"):
            continue
        metrics_present[m] = True
    return all(metrics_present.values())


def coerce_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def build_summary(rows: List[Dict[str, str]]):
    # Group by part_number
    by_pn: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        pn = r.get("part_number") or ""
        by_pn.setdefault(pn, []).append(r)

    summary = []
    for pn, part_rows in by_pn.items():
        if not fully_matched(part_rows):
            continue
        rec: Dict[str, Optional[float]] = {
            "part_number": pn,
            "repeatability_value_um": None,
            "repeatability_spec_um": None,
            "repeatability_pass": None,
            "uncal_value_um": None,
            "uncal_spec_um": None,
            "uncal_pass": None,
            "cal_value_um": None,
            "cal_spec_um": None,
            "cal_pass": None,
        }
        pass_all = True
        for r in part_rows:
            metric = r.get("metric")
            val = coerce_float(r.get("value"))
            spec = coerce_float(r.get("spec_max"))
            p = r.get("pass")
            if metric == "Average_Repeatability":
                rec["repeatability_value_um"] = val
                rec["repeatability_spec_um"] = spec
                rec["repeatability_pass"] = (p == "True")
            elif metric == "Avg_Uncalibrated_Accuracy":
                rec["uncal_value_um"] = val
                rec["uncal_spec_um"] = spec
                rec["uncal_pass"] = (p == "True")
            elif metric == "Avg_Calibrated_Accuracy":
                rec["cal_value_um"] = val
                rec["cal_spec_um"] = spec
                rec["cal_pass"] = (p == "True")
        for k in ("repeatability_pass", "uncal_pass", "cal_pass"):
            pass_all = pass_all and bool(rec[k])
        rec["pass_all"] = pass_all
        summary.append(rec)
    # Sort by part_number
    summary.sort(key=lambda x: x["part_number"])
    return summary


def write_summary_csv(items, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "part_number",
        "pass_all",
        "repeatability_value_um",
        "repeatability_spec_um",
        "repeatability_pass",
        "uncal_value_um",
        "uncal_spec_um",
        "uncal_pass",
        "cal_value_um",
        "cal_spec_um",
        "cal_pass",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for it in items:
            w.writerow(it)


def write_summary_html(items, out_html: Path):
    out_html.parent.mkdir(parents=True, exist_ok=True)
    data_json = json.dumps(items)
    html = """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Parts Performance Summary</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 20px; }
    label { font-weight: 600; }
    select { min-width: 360px; padding: 6px; }
    table { border-collapse: collapse; margin-top: 16px; }
    th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; }
    .pass { color: #0a7f24; font-weight: 600; }
    .fail { color: #b00020; font-weight: 600; }
  </style>
</head>
<body>
  <h1>Parts Performance Summary</h1>
  <label for=\"pn\">Select part number</label><br/>
  <select id=\"pn\"></select>
  <div id=\"details\"></div>
  <script>
    const items = __DATA_JSON__;
    const pnSel = document.getElementById('pn');
    const details = document.getElementById('details');

    const byPn = new Map(items.map(x => [x.part_number, x]));
    for (const pn of [...byPn.keys()].sort()) {
      const opt = document.createElement('option');
      opt.value = pn; opt.textContent = pn;
      pnSel.appendChild(opt);
    }

    function render(pn) {
      const x = byPn.get(pn);
      if (!x) { details.innerHTML=''; return; }
      const tf = v => v ? 'True' : 'False';
      const cls = v => v ? 'pass' : 'fail';
      details.innerHTML = `
        <h2>${pn} — Overall: <span class=\"${cls(x.pass_all)}\">${tf(x.pass_all)}</span></h2>
        <table>
          <thead><tr><th>Metric</th><th>Value (µm)</th><th>Spec Max (µm)</th><th>Pass</th></tr></thead>
          <tbody>
            <tr><td>Average_Repeatability</td><td>${x.repeatability_value_um || ''}</td><td>${x.repeatability_spec_um || ''}</td><td class=\"${cls(x.repeatability_pass)}\">${tf(x.repeatability_pass)}</td></tr>
            <tr><td>Avg_Uncalibrated_Accuracy</td><td>${x.uncal_value_um || ''}</td><td>${x.uncal_spec_um || ''}</td><td class=\"${cls(x.uncal_pass)}\">${tf(x.uncal_pass)}</td></tr>
            <tr><td>Avg_Calibrated_Accuracy</td><td>${x.cal_value_um || ''}</td><td>${x.cal_spec_um || ''}</td><td class=\"${cls(x.cal_pass)}\">${tf(x.cal_pass)}</td></tr>
          </tbody>
        </table>
      `;
    }

    pnSel.addEventListener('change', e => render(e.target.value));
    if (pnSel.options.length) { pnSel.selectedIndex = 0; render(pnSel.value); }
  </script>
</body>
</html>
"""
    html = html.replace("__DATA_JSON__", data_json)
    out_html.write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Build a summary from report.csv (fully matched PNs only)")
    ap.add_argument("--report", required=True, help="Path to report CSV produced by report.py")
    ap.add_argument("--out-csv", default=str(Path("reports")/"summary.csv"))
    ap.add_argument("--out-html", default=str(Path("reports")/"summary.html"))
    args = ap.parse_args()

    rows = read_rows(Path(args.report))
    items = build_summary(rows)
    write_summary_csv(items, Path(args.out_csv))
    write_summary_html(items, Path(args.out_html))
    print(f"PNs summarized: {len(items)}")
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_html}")


if __name__ == "__main__":
    main()
