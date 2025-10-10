#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import importlib.util
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------
# Original simple spec/tests pipeline (sample files)
# -------------------------
def normalize_specs(data: Any) -> Dict[str, Dict[str, Dict[str, float]]]:
    # Accept either {"parts": {PN: {metric: {min,max,equals}}}} or {PN: {...}}
    parts = data.get("parts") if isinstance(data, dict) else None
    if parts is None and isinstance(data, dict):
        parts = data
    if not isinstance(parts, dict):
        raise ValueError("Specs JSON must be a mapping of part numbers to metric constraints")
    normalized: Dict[str, Dict[str, Dict[str, float]]] = {}
    for pn, metrics in parts.items():
        if not isinstance(metrics, dict):
            raise ValueError(f"Specs for part {pn} must be an object of metrics")
        normalized[pn] = {}
        for metric, constraint in metrics.items():
            if not isinstance(constraint, dict):
                raise ValueError(f"Constraint for {pn}.{metric} must be an object")
            allowed = {k: constraint[k] for k in ("min", "max", "equals") if k in constraint}
            if not allowed:
                raise ValueError(f"Constraint for {pn}.{metric} must include min/max/equals")
            normalized[pn][metric] = allowed
    return normalized


def normalize_tests(data: Any) -> List[Dict[str, Any]]:
    # Accept a list of entries or a mapping of part_number -> list/results
    entries: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            entries.append(item)
    elif isinstance(data, dict):
        for pn, val in data.items():
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        item = {**item}
                        item.setdefault("part_number", pn)
                        entries.append(item)
            elif isinstance(val, dict):
                entries.append({"part_number": pn, **val})
    else:
        raise ValueError("Tests JSON must be a list or a mapping")

    # Ensure structure {part_number, unit_id?, results:{metric:value}}
    norm: List[Dict[str, Any]] = []
    for e in entries:
        pn = e.get("part_number")
        results = e.get("results") if isinstance(e.get("results"), dict) else None
        if results is None:
            # try flatten
            results = {k: v for k, v in e.items() if k not in ("part_number", "unit_id") and isinstance(v, (int, float))}
        if not pn or not isinstance(results, dict) or not results:
            # skip invalid entries
            continue
        norm.append({
            "part_number": pn,
            "unit_id": e.get("unit_id"),
            "results": results,
        })
    return norm


def evaluate(value: float, spec: Dict[str, float]) -> Tuple[bool, Optional[float], Optional[float]]:
    # returns (pass, delta_to_min, delta_to_max)
    if "equals" in spec:
        ok = (value == spec["equals"])  # exact match
        return ok, None, None
    d_min = None
    d_max = None
    ok = True
    if "min" in spec:
        d_min = value - float(spec["min"])  # positive means above min
        if d_min < 0:
            ok = False
    if "max" in spec:
        d_max = float(spec["max"]) - value  # positive means below max
        if d_max < 0:
            ok = False
    return ok, d_min, d_max


def generate_rows(tests: List[Dict[str, Any]], specs: Dict[str, Dict[str, Dict[str, float]]]) -> Iterable[Dict[str, Any]]:
    for entry in tests:
        pn = entry["part_number"]
        unit = entry.get("unit_id")
        results: Dict[str, Any] = entry["results"]
        part_specs = specs.get(pn)
        if part_specs is None:
            # Unknown part: report all metrics without specs
            for metric, value in results.items():
                yield {
                    "part_number": pn,
                    "unit_id": unit,
                    "metric": metric,
                    "value": value,
                    "spec_min": None,
                    "spec_max": None,
                    "pass": None,
                    "delta_to_min": None,
                    "delta_to_max": None,
                    "note": "No specs for part_number",
                }
            continue
        for metric, value in results.items():
            spec = part_specs.get(metric)
            if spec is None:
                yield {
                    "part_number": pn,
                    "unit_id": unit,
                    "metric": metric,
                    "value": value,
                    "spec_min": None,
                    "spec_max": None,
                    "pass": None,
                    "delta_to_min": None,
                    "delta_to_max": None,
                    "note": "No spec for metric",
                }
            else:
                try:
                    v = float(value)
                except Exception:
                    yield {
                        "part_number": pn,
                        "unit_id": unit,
                        "metric": metric,
                        "value": value,
                        "spec_min": spec.get("min"),
                        "spec_max": spec.get("max"),
                        "pass": False,
                        "delta_to_min": None,
                        "delta_to_max": None,
                        "note": "Non-numeric value",
                    }
                    continue
                ok, dmin, dmax = evaluate(v, spec)
                yield {
                    "part_number": pn,
                    "unit_id": unit,
                    "metric": metric,
                    "value": v,
                    "spec_min": spec.get("min"),
                    "spec_max": spec.get("max"),
                    "pass": ok,
                    "delta_to_min": dmin,
                    "delta_to_max": dmax,
                    "note": None,
                }


def write_csv(rows: Iterable[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fieldnames = [
        "part_number", "unit_id", "metric", "value",
        "spec_min", "spec_max", "pass", "delta_to_min", "delta_to_max", "note",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# -------------------------
# Product Specs + Results (real data) pipeline
# -------------------------
CANON_UP = lambda s: s.strip().upper()
ZEROPAD_RE = re.compile(r"-(0+)(\d+)\b")
ENCODER_ORDER_RE = re.compile(r"-(?:E\d+|ES\d+)\b", re.IGNORECASE)
OPTIONAL_CTRL_TAGS = ("-ACS", "-ASR")  # not stripped by default


def normalize_zeropad(s: str) -> str:
    # Collapse hyphenated zero-padded numeric suffix, e.g., -0100 -> -100
    return ZEROPAD_RE.sub(r"-\2", s)


def strip_parenthetical(s: str) -> Tuple[str, bool]:
    idx = s.find("(")
    if idx >= 0:
        return s[:idx].rstrip(), True
    return s, False


def strip_encoder_order(s: str) -> Tuple[str, bool]:
    new_s, n = ENCODER_ORDER_RE.subn("", s)
    if n > 0:
        new_s = re.sub(r"--+", "-", new_s).rstrip("-")
        return new_s, True
    return s, False


def longest_base_prefix(s: str, bases: List[str]) -> Optional[str]:
    hits = [b for b in bases if s.startswith(b)]
    if not hits:
        return None
    return max(hits, key=len)


def build_product_specs_index(specs: Any) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
    Flattens Product Specs.json to {PN(upper, zero-pad-normalized): spec_obj_with_mech}, returns (index, base_models_upper_list)
    spec_obj_with_mech keeps original structure so we can read mechanical specs.
    """
    index: Dict[str, Dict[str, Any]] = {}
    base_models: List[str] = []
    if not isinstance(specs, dict):
        return index, base_models
    for base, entries in specs.items():
        base_models.append(CANON_UP(base))
        if isinstance(entries, list):
            for item in entries:
                if isinstance(item, dict):
                    for pn, payload in item.items():
                        key = CANON_UP(normalize_zeropad(pn))
                        index[key] = payload
        elif isinstance(entries, dict):
            for pn, payload in entries.items():
                key = CANON_UP(normalize_zeropad(pn))
                index[key] = payload
    base_models.sort(key=len, reverse=True)
    return index, base_models


def classify_map_to_spec_pn(raw_pn: str, spec_index: Dict[str, Dict[str, Any]], base_models: List[str], allow_ctrl_strip: bool = False) -> Tuple[str, Optional[str], Optional[str], str]:
    """
    Returns (match_type, matched_spec_pn_key_norm, matched_base_model, reason)
    matched_spec_pn_key_norm is the canonical key (upper + zero-pad normalized)
    """
    reason_parts: List[str] = []
    pn0 = CANON_UP(raw_pn)
    pn_norm = CANON_UP(normalize_zeropad(raw_pn))
    if pn_norm in spec_index:
        base = longest_base_prefix(pn_norm, base_models)
        return "normalized", pn_norm, base, "zero-pad and case-insensitive match"
    p1, sp = strip_parenthetical(pn0)
    p2, se = strip_encoder_order(p1)
    if sp:
        reason_parts.append("removed parenthetical tail")
    if se:
        reason_parts.append("stripped -E/-ES suffixes")
    p3 = p2
    if allow_ctrl_strip:
        for tag in OPTIONAL_CTRL_TAGS:
            if p3.endswith(tag):
                p3 = p3[: -len(tag)].rstrip("-")
                reason_parts.append(f"stripped {tag}")
                break
    p3 = CANON_UP(normalize_zeropad(p3))
    if p3 in spec_index:
        base = longest_base_prefix(p3, base_models)
        return "stripped", p3, base, "; ".join(reason_parts) or "stripped to match"
    base = longest_base_prefix(p3, base_models)
    if base:
        return "base-only", None, base, "no specific PN under base in specs"
    return "no-base", None, None, "no base model found in specs"


def parse_numeric_with_unit(s: str) -> Tuple[Optional[float], Optional[str], bool]:
    """Extract the largest numeric value and a coarse unit token from a spec string.
    Returns (number, unit, has_plus_minus) or (None, None, False) if not parseable.
    Recognized units: um/µm, nm, mm, arcsec, urad (others ignored for our metrics).
    """
    if not isinstance(s, str):
        return None, None, False
    # Find numbers (including decimals)
    nums = [float(x) for x in re.findall(r"[-+]?(?:\d+\.\d+|\d+)", s)]
    unit = None
    s_low = s.lower()
    if ("µm" in s) or (" um" in s_low) or (" micrometer" in s_low) or (" micron" in s_low):
        unit = "um"
    elif re.search(r"\bnm\b", s_low):
        unit = "nm"
    elif re.search(r"\bmm\b", s_low):
        unit = "mm"
    elif "arc sec" in s_low or "arcsec" in s_low or re.search(r"\barc\b", s_low):
        unit = "arcsec"
    elif "µrad" in s or "urad" in s_low or re.search(r"\brad\b", s_low):
        unit = "urad"
    # Detect plus-minus encoding: ± or Â± or '+/-'
    has_pm = ('±' in s) or ('Â±' in s) or ('+/-' in s_low)
    if not nums:
        return None, unit, has_pm
    # choose the largest value (worst-case tolerance)
    return max(nums), unit, has_pm


def best_mech_limit(mech: Dict[str, Any], want: str, allow_generic_for_standard: bool = False) -> Tuple[Optional[float], Optional[str], Optional[str], bool]:
    """Search mechanical specs for best numeric limit for a wanted class, using explicit keyword rules.
    want in {"repeatability", "accuracy_standard", "accuracy_calibrated"}
    Returns (limit_value, unit, matched_key)
    """
    if not isinstance(mech, dict):
        return None, None, None, False

    def collect(filter_fn) -> List[Tuple[float, str, str, bool, Optional[float]]]:
        out: List[Tuple[float, str, str, bool, Optional[float]]] = []
        def to_um_local(val: float, u: Optional[str]) -> Optional[float]:
            if u in (None, "", "um"):
                return val
            if u == "nm":
                return val / 1000.0
            if u == "mm":
                return val * 1000.0
            return None
        for k, v in mech.items():
            if not isinstance(v, str):
                continue
            kl = k.lower()
            if not filter_fn(kl):
                continue
            num, unit, has_pm = parse_numeric_with_unit(v)
            if num is None:
                continue
            norm = to_um_local(num, unit or None)
            out.append((num, (unit or ""), k, has_pm, norm))
        return out

    candidates: List[Tuple[float, str, str, bool, Optional[float]]] = []
    if want == "repeatability":
        # Prefer keys that include both "repeatability" and "bidirectional" (keywords, not exact match)
        candidates = collect(lambda kl: ("repeatability" in kl and "bidirectional" in kl))
        if not candidates:
            # Fallback to any repeatability if bidirectional form not found
            candidates = collect(lambda kl: ("repeatability" in kl))
    elif want == "accuracy_standard":
        # Non-calibrated accuracy must explicitly indicate Standard/Uncalibrated/Base; do NOT use generic accuracy here
        candidates = collect(lambda kl: ("accuracy" in kl and "calibrated" not in kl and ("standard" in kl or "uncalibrated" in kl or "base" in kl) and ("plus" not in kl)))
        if not candidates and allow_generic_for_standard:
            # For families like PlanarDL with PL1, allow generic 'Accuracy' (non-calibrated) when explicitly using PL1
            candidates = collect(lambda kl: ("accuracy" in kl and "calibrated" not in kl and ("plus" not in kl)))
    elif want == "accuracy_calibrated":
        # Accuracy that explicitly mentions Calibrated/Plus
        candidates = collect(lambda kl: ("accuracy" in kl and ("calibrated" in kl or "plus" in kl)))
        if not candidates:
            # If there is no explicit Calibrated/Plus accuracy, assume provided accuracy is calibrated
            candidates = collect(lambda kl: ("accuracy" in kl and "calibrated" not in kl))
    else:
        candidates = []

    if not candidates:
        return None, None, None, False

    # Choose the candidate with the maximum numeric tolerance (most permissive bound)
    # Prefer the candidate with the largest normalized µm value; fallback to raw value if normalization missing
    num, unit, key, has_pm, norm = max(candidates, key=lambda t: (t[4] if t[4] is not None else t[0]))
    return num, (unit or None), key, has_pm


def process_product_specs_mode(results_raw: Any, specs_raw: Any, allow_ctrl_strip: bool = False) -> Iterable[Dict[str, Any]]:
    """Process real Product Specs.json and Results.json to yield report rows."""
    spec_index, base_models = build_product_specs_index(specs_raw)

    if not isinstance(results_raw, list):
        raise ValueError("Results JSON must be a list of objects with BasePartNum and averages")

    def get_mech_for_key(key_norm: Optional[str]):
        if key_norm is None:
            return None, None
        payload = spec_index.get(key_norm) or {}
        specs_obj = payload.get("specifications") if isinstance(payload, dict) else None
        mech_local = specs_obj.get("mechanical") if isinstance(specs_obj, dict) else None
        return mech_local, key_norm

    for rec in results_raw:
        if not isinstance(rec, dict):
            continue
        base_pn = rec.get("BasePartNum")
        if not base_pn:
            continue
        mtype, matched_key, base, reason = classify_map_to_spec_pn(str(base_pn), spec_index, base_models, allow_ctrl_strip=allow_ctrl_strip)

        # Collect values (as floats when possible)
        def fget(name: str) -> Optional[float]:
            v = rec.get(name)
            try:
                if v is None:
                    return None
                return float(v)
            except Exception:
                return None

        avg_rep = fget("Average_Repeatability")
        avg_acc_uncal = fget("Avg_Uncalibrated_Accuracy")
        avg_acc_cal = fget("Avg_Calibrated_Accuracy")

        mech = None
        matched_spec_pn_display = None
        if matched_key is not None:
            mech, matched_spec_pn_display = get_mech_for_key(matched_key)
        
        # Helper: try PL1/PL2 variants when we only have a base match or mech is None
        pn_norm = CANON_UP(normalize_zeropad(str(base_pn)))
        def mech_with_pl(want: str):
            # Prefer PL1 for uncalibrated, PL2 for calibrated, for repeatability try PL2 then PL1
            choices = []
            if want == "accuracy_standard":
                choices = [pn_norm + "-PL1"]
            elif want == "accuracy_calibrated":
                # Consider PL2/PL3/PL4 as calibrated performance levels
                choices = [pn_norm + "-PL2", pn_norm + "-PL3", pn_norm + "-PL4"]
            else:
                # Repeatability: try calibrated variants first, then base
                choices = [pn_norm + "-PL2", pn_norm + "-PL3", pn_norm + "-PL4", pn_norm + "-PL1"]
            for key in choices:
                m, disp = get_mech_for_key(key)
                if m is not None:
                    return m, disp
            return None, None

        def emit(metric: str, value: Optional[float], want: str):
            if value is None:
                return
            # Results are assumed to be in microns (um)
            value_um = value

            local_mech = mech
            local_pn_disp = matched_spec_pn_display or (pn_norm)
            if local_mech is None and (mtype == "base-only" or matched_key is None):
                alt_mech, alt_disp = mech_with_pl(want)
                if alt_mech is not None:
                    local_mech = alt_mech
                    local_pn_disp = alt_disp or local_pn_disp

            if local_mech is None:
                note = "No specific PN in specs" if mtype == "base-only" else reason or "No specs"
                yield {
                    "part_number": local_pn_disp,
                    "unit_id": None,
                    "metric": metric,
                    "value": value_um,
                    "spec_min": None,
                    "spec_max": None,
                    "pass": None,
                    "delta_to_min": None,
                    "delta_to_max": None,
                    "note": note,
                }
                return
            # If we derived PL1 explicitly for uncalibrated accuracy, allow generic accuracy keys
            allow_generic = False
            if want == "accuracy_standard":
                disp_upper = (local_pn_disp or "").upper()
                if disp_upper.endswith("-PL1"):
                    allow_generic = True
            limit, unit, key, has_pm = best_mech_limit(local_mech, want, allow_generic_for_standard=allow_generic)
            if limit is None:
                yield {
                    "part_number": matched_spec_pn_display or str(base_pn),
                    "unit_id": None,
                    "metric": metric,
                    "value": value_um,
                    "spec_min": None,
                    "spec_max": None,
                    "pass": None,
                    "delta_to_min": None,
                    "delta_to_max": None,
                    "note": "No matching spec key",
                }
                return
            # Convert spec to microns where possible
            def to_um(val: float, u: Optional[str]) -> Optional[float]:
                if u in (None, "", "um"):
                    return val
                if u == "nm":
                    return val / 1000.0
                if u == "mm":
                    return val * 1000.0
                return None  # unsupported units for this metric
            # If spec is "+/- X", convert to peak-to-peak by doubling for repeatability and accuracy
            if has_pm and limit is not None and want in ("repeatability", "accuracy_standard", "accuracy_calibrated"):
                limit = limit * 2.0
            limit_um = to_um(limit, unit)
            if limit_um is None:
                yield {
                    "part_number": matched_spec_pn_display or str(base_pn),
                    "unit_id": None,
                    "metric": metric,
                    "value": value_um,
                    "spec_min": None,
                    "spec_max": None,
                    "pass": None,
                    "delta_to_min": None,
                    "delta_to_max": None,
                    "note": f"Unit mismatch (spec {unit})",
                }
                return
            ok = value_um <= limit_um
            yield {
                "part_number": matched_spec_pn_display or str(base_pn),
                "unit_id": None,
                "metric": metric,
                "value": value_um,
                "spec_min": None,
                "spec_max": limit_um,
                "pass": ok,
                "delta_to_min": None,
                "delta_to_max": (limit_um - value_um),
                "note": None if ok else f"Exceeded {key} ({limit_um} um)",
            }

        # Emit three metrics if present
        yield from emit("Average_Repeatability", avg_rep, "repeatability")
        yield from emit("Avg_Uncalibrated_Accuracy", avg_acc_uncal, "accuracy_standard")
        yield from emit("Avg_Calibrated_Accuracy", avg_acc_cal, "accuracy_calibrated")


def looks_like_product_specs(specs_raw: Any) -> bool:
    if not isinstance(specs_raw, dict):
        return False
    if "parts" in specs_raw:
        return False
    # Heuristic: any value is list of dicts with inner object having 'specifications'
    for v in specs_raw.values():
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    for inner in item.values():
                        if isinstance(inner, dict) and "specifications" in inner:
                            return True
    return False


def write_csv(rows: Iterable[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fieldnames = [
        "part_number", "unit_id", "metric", "value",
        "spec_min", "spec_max", "pass", "delta_to_min", "delta_to_max", "note",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _run_summary(report_path: str, out_csv: Optional[str], out_html: Optional[str]) -> None:
    try:
        base_dir = Path(__file__).parent
        mod_path = base_dir / "scripts" / "build_summary_ui.py"
        spec = importlib.util.spec_from_file_location("build_summary_ui", str(mod_path))
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = module.read_rows(Path(report_path))
        items = module.build_summary(rows)
        module.write_summary_csv(items, Path(out_csv or Path(report_path).with_name("summary.csv")))
        module.write_summary_html(items, Path(out_html or Path(report_path).with_name("summary.html")))
    except Exception:
        # Best-effort; do not raise if summary generation fails
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Report performance vs specs.")
    parser.add_argument("--tests", required=True, help="Path to JSON with test results (Results.json or sample)")
    parser.add_argument("--specs", required=True, help="Path to JSON with specs (Product Specs.json or sample)")
    parser.add_argument("--out", default="report.csv", help="Output CSV path")
    parser.add_argument("--allow-ctrl-strip", action="store_true", help="Allow stripping trailing -ACS/-ASR when mapping part numbers")
    parser.add_argument("--summary", action="store_true", help="Also generate summary.csv and summary.html for fully matched PNs")
    parser.add_argument("--summary-csv", default=None, help="Optional path for summary CSV output")
    parser.add_argument("--summary-html", default=None, help="Optional path for summary HTML output")
    args = parser.parse_args()

    tests_raw = load_json(args.tests)
    specs_raw = load_json(args.specs)

    if looks_like_product_specs(specs_raw):
        rows = list(process_product_specs_mode(tests_raw, specs_raw, allow_ctrl_strip=args.allow_ctrl_strip))
        write_csv(rows, args.out)
        if args.summary:
            _run_summary(args.out, args.summary_csv, args.summary_html)
        return

    # Fallback to original simple flow (sample files)
    tests = normalize_tests(tests_raw)
    specs = normalize_specs(specs_raw)
    rows = list(generate_rows(tests, specs))
    write_csv(rows, args.out)
    if args.summary:
        _run_summary(args.out, args.summary_csv, args.summary_html)


if __name__ == "__main__":
    main()
