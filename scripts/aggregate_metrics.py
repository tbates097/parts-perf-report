#!/usr/bin/env python3
import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def to_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        try:
            return float(s)
        except Exception:
            return None
    s = str(s).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def to_bool(s: Optional[str]) -> Optional[bool]:
    if s is None:
        return None
    s = str(s).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    return None


def read_report_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def aggregate(report_csv: Path, out_dir: Path, top_n: int = 20) -> Dict[str, Path]:
    rows = read_report_rows(report_csv)

    # Filter to evaluable rows (with concrete pass True/False and spec_max)
    eval_rows: List[Tuple[str, str, Optional[float], Optional[float], Optional[bool]]] = []
    for r in rows:
        pn = (r.get("part_number") or "").strip()
        metric = (r.get("metric") or "").strip()
        val = to_float(r.get("value"))
        smax = to_float(r.get("spec_max"))
        p = to_bool(r.get("pass"))
        if pn and metric and (p is not None) and (smax is not None) and (val is not None):
            eval_rows.append((pn, metric, val, smax, p))

    by_pn: Dict[str, List[Tuple[str, float, float, bool]]] = {}
    by_pn_metric: Dict[Tuple[str, str], List[Tuple[float, float, bool]]] = {}

    for pn, metric, val, smax, p in eval_rows:
        by_pn.setdefault(pn, []).append((metric, val, smax, p))
        key = (pn, metric)
        by_pn_metric.setdefault(key, []).append((val, smax, p))

    per_part_rows: List[Dict[str, object]] = []
    top_fail_rate_rows: List[Dict[str, object]] = []
    top_worst_margin_rows: List[Dict[str, object]] = []

    overall_eval = 0
    overall_pass = 0

    for pn, items in by_pn.items():
        evaluable = len(items)
        passes = sum(1 for _, _, _, p in items if p)
        fails = evaluable - passes
        pass_rate = (passes / evaluable) if evaluable else None
        margins = [(smax - val) for _, val, smax, _ in items]
        worst_margin = min(margins) if margins else None  # most negative is worst
        avg_margin = (sum(margins) / len(margins)) if margins else None
        worst_idx = None
        if margins:
            m = min((margins[i], i) for i in range(len(margins)))
            worst_idx = m[1]
        worst_metric = items[worst_idx][0] if worst_idx is not None else None
        worst_value = items[worst_idx][1] if worst_idx is not None else None
        worst_spec = items[worst_idx][2] if worst_idx is not None else None

        per_part_rows.append({
            "part_number": pn,
            "evaluable": evaluable,
            "pass_count": passes,
            "fail_count": fails,
            "pass_rate": round(pass_rate, 4) if pass_rate is not None else None,
            "avg_margin_um": round(avg_margin, 4) if avg_margin is not None else None,
            "worst_margin_um": round(worst_margin, 4) if worst_margin is not None else None,
            "worst_metric": worst_metric,
            "worst_value_um": round(worst_value, 4) if worst_value is not None else None,
            "worst_spec_um": round(worst_spec, 4) if worst_spec is not None else None,
        })

        overall_eval += evaluable
        overall_pass += passes

    # Per metric per part
    per_metric_part_rows: List[Dict[str, object]] = []
    for (pn, metric), items in by_pn_metric.items():
        evaluable = len(items)
        passes = sum(1 for _, _, p in items if p)
        fails = evaluable - passes
        pass_rate = (passes / evaluable) if evaluable else None
        margins = [(smax - val) for val, smax, _ in items]
        worst_margin = min(margins) if margins else None
        per_metric_part_rows.append({
            "part_number": pn,
            "metric": metric,
            "evaluable": evaluable,
            "pass_count": passes,
            "fail_count": fails,
            "pass_rate": round(pass_rate, 4) if pass_rate is not None else None,
            "worst_margin_um": round(worst_margin, 4) if worst_margin is not None else None,
        })

    # Top offenders by fail rate (min threshold of evaluable to avoid noise)
    min_eval_threshold = 3
    offenders_fail = [r for r in per_part_rows if r["evaluable"] >= min_eval_threshold]
    offenders_fail.sort(key=lambda r: (-(r["fail_count"]/r["evaluable"] if r["evaluable"] else 0), -r["evaluable"]))
    top_fail_rate_rows = offenders_fail[:top_n]

    # Top offenders by worst (most negative) margin
    offenders_margin = [r for r in per_part_rows if r["worst_margin_um"] is not None]
    offenders_margin.sort(key=lambda r: (r["worst_margin_um"]))  # ascending; most negative first
    top_worst_margin_rows = offenders_margin[:top_n]

    # Overall summary
    overall_pass_rate = (overall_pass / overall_eval) if overall_eval else None
    overall_rows = [{
        "overall_evaluable": overall_eval,
        "overall_pass": overall_pass,
        "overall_fail": overall_eval - overall_pass,
        "overall_pass_rate": round(overall_pass_rate, 4) if overall_pass_rate is not None else None,
    }]

    # Write outputs
    out = {}
    per_part_path = out_dir / "aggregates" / "per_part.csv"
    write_csv(per_part_path, per_part_rows, [
        "part_number","evaluable","pass_count","fail_count","pass_rate","avg_margin_um","worst_margin_um","worst_metric","worst_value_um","worst_spec_um"
    ])
    out["per_part"] = per_part_path

    per_metric_part_path = out_dir / "aggregates" / "per_metric_per_part.csv"
    write_csv(per_metric_part_path, per_metric_part_rows, [
        "part_number","metric","evaluable","pass_count","fail_count","pass_rate","worst_margin_um"
    ])
    out["per_metric_per_part"] = per_metric_part_path

    top_fail_path = out_dir / "aggregates" / "top_offenders_by_fail_rate.csv"
    write_csv(top_fail_path, top_fail_rate_rows, [
        "part_number","evaluable","pass_count","fail_count","pass_rate","worst_margin_um","worst_metric"
    ])
    out["top_offenders_by_fail_rate"] = top_fail_path

    top_margin_path = out_dir / "aggregates" / "top_offenders_by_worst_margin.csv"
    write_csv(top_margin_path, top_worst_margin_rows, [
        "part_number","evaluable","pass_count","fail_count","pass_rate","worst_margin_um","worst_metric","worst_value_um","worst_spec_um"
    ])
    out["top_offenders_by_worst_margin"] = top_margin_path

    overall_path = out_dir / "aggregates" / "overall.csv"
    write_csv(overall_path, overall_rows, [
        "overall_evaluable","overall_pass","overall_fail","overall_pass_rate"
    ])
    out["overall"] = overall_path

    # Try to generate charts if matplotlib is available
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
        charts_dir = out_dir / "charts"
        ensure_dir(charts_dir)

        # Top N fail rate chart
        labels = [r["part_number"] for r in top_fail_rate_rows]
        values = [(r["fail_count"]/r["evaluable"]) if r["evaluable"] else 0 for r in top_fail_rate_rows]
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(labels)), values, color="#c62828")
        plt.xticks(range(len(labels)), labels, rotation=60, ha='right')
        plt.ylabel("Fail rate")
        plt.title(f"Top {min(top_n, len(labels))} offenders by fail rate")
        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f"{v:.2%}", ha='center', va='bottom', fontsize=8)
        fail_chart_path = charts_dir / "top_offenders_fail_rate.png"
        plt.tight_layout()
        plt.savefig(fail_chart_path, dpi=150)
        plt.close()
        out["chart_top_fail_rate"] = fail_chart_path

        # Top N worst margin chart (convert to positive exceedance magnitude)
        labels2 = [r["part_number"] for r in top_worst_margin_rows]
        margins2 = [r["worst_margin_um"] for r in top_worst_margin_rows]
        exceed_mag = [(-m if (m is not None and m < 0) else 0) for m in margins2]
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(labels2)), exceed_mag, color="#6a1b9a")
        plt.xticks(range(len(labels2)), labels2, rotation=60, ha='right')
        plt.ylabel("Worst exceedance (µm)")
        plt.title(f"Top {min(top_n, len(labels2))} offenders by worst exceedance")
        for i, v in enumerate(exceed_mag):
            plt.text(i, v + max(exceed_mag)*0.01 if exceed_mag else 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
        worst_chart_path = charts_dir / "top_offenders_worst_margin.png"
        plt.tight_layout()
        plt.savefig(worst_chart_path, dpi=150)
        plt.close()
        out["chart_top_worst_margin"] = worst_chart_path

    except Exception:
        # Charts optional; ignore failures (e.g., matplotlib not installed)
        pass

    # Produce a Slides outline for easy copy/paste into Google Slides
    outline_lines = [
        "Title: Parts Performance Summary",
        "Subtitle: Pass rates, top offenders, and key metrics",
        "",
        "Slide 1: Executive Summary",
        "- Overall pass rate: see overall.csv",
        "- Top offenders (fail rate): see charts/top_offenders_fail_rate.png",
        "- Top offenders (worst exceedance): see charts/top_offenders_worst_margin.png",
        "",
        "Slide 2: Overall Metrics",
        "- Table: overall.csv",
        "- Note on methodology: rows require concrete spec and pass/fail",
        "",
        "Slide 3: Top Offenders by Fail Rate",
        "- Image: charts/top_offenders_fail_rate.png",
        "- Table: aggregates/top_offenders_by_fail_rate.csv",
        "",
        "Slide 4: Top Offenders by Worst Exceedance",
        "- Image: charts/top_offenders_worst_margin.png",
        "- Table: aggregates/top_offenders_by_worst_margin.csv",
        "",
        "Slide 5+: By-Part Detail (as needed)",
        "- Table: aggregates/per_part.csv",
        "- Table: aggregates/per_metric_per_part.csv",
    ]
    outline_path = out_dir / "slides" / "outline.txt"
    ensure_dir(outline_path.parent)
    outline_path.write_text("\n".join(outline_lines), encoding="utf-8")

    out["slides_outline"] = outline_path
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate pass/fail metrics and biggest offenders from report.csv")
    ap.add_argument("--report", required=True, help="Path to reports/report.csv produced by report.py")
    ap.add_argument("--out-dir", default=str(Path("reports")), help="Output directory for aggregates and charts")
    ap.add_argument("--top", type=int, default=20, help="Top N to include in offender charts/tables")
    args = ap.parse_args()

    report_csv = Path(args.report)
    out_dir = Path(args.out_dir)
    res = aggregate(report_csv, out_dir, top_n=int(args.top))
    # Print minimal pointers
    print(f"Wrote {res.get('per_part')}")
    print(f"Wrote {res.get('per_metric_per_part')}")
    print(f"Wrote {res.get('top_offenders_by_fail_rate')}")
    print(f"Wrote {res.get('top_offenders_by_worst_margin')}")
    print(f"Wrote {res.get('overall')}")
    if 'chart_top_fail_rate' in res:
        print(f"Wrote {res.get('chart_top_fail_rate')}")
    if 'chart_top_worst_margin' in res:
        print(f"Wrote {res.get('chart_top_worst_margin')}")
    print(f"Wrote {res.get('slides_outline')}")


if __name__ == "__main__":
    main()