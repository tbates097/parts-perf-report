#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

# Conservative normalization rules:
# - Preserve meaningful family/variant designators (LM, SL, SLE, L, LZ, LZS, XY, Z, WB, B, etc.)
# - Case-insensitive comparisons
# - Normalize zero-padded numeric suffixes: -0NNN -> -NNN
# - Strip only trailing encoder/order codes not present in spec PN keys:
#   * -E<digits>
#   * -ES<digits>
# - Strip any parenthetical suffixes: e.g., "... (ANT95" or "... (" -> strip from '(' to end
# - Do NOT strip tokens like -SLE, -WB, -Z, -XY, -LZ, -LZS by default
# - Optionally allow stripping of -ACS/-ASR via a flag, but default is off

KEEP_TOKENS_EXAMPLES = {
    "LM", "SL", "SLE", "L", "LZ", "LZS", "XY", "Z", "WB", "B"
}

ENCODER_ORDER_RE = re.compile(r"-(?:E\d+|ES\d+)\b", re.IGNORECASE)
ZEROPAD_RE = re.compile(r"-(0+)(\d+)\b")

OPTIONAL_CTRL_TAGS = ("-ACS", "-ASR")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_zeropad(s: str) -> str:
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


def canonical(s: str) -> str:
    return s.strip().upper()


def longest_base_prefix(s: str, bases: List[str]) -> Optional[str]:
    # Return the longest base from bases that is a prefix of s
    hits = [b for b in bases if s.startswith(b)]
    if not hits:
        return None
    return max(hits, key=len)


def build_spec_index(specs: dict) -> Tuple[Dict[str, str], List[str]]:
    # Return mapping: spec_norm_key -> original_spec_pn, and list of base models (uppercased)
    spec_pn_map: Dict[str, str] = {}
    base_models: List[str] = []
    for base, entries in specs.items():
        base_models.append(canonical(base))
        if isinstance(entries, list):
            for item in entries:
                if isinstance(item, dict):
                    for pn in item.keys():
                        pn_can = canonical(normalize_zeropad(pn))
                        spec_pn_map[pn_can] = pn  # original form
        elif isinstance(entries, dict):
            for pn in entries.keys():
                pn_can = canonical(normalize_zeropad(pn))
                spec_pn_map[pn_can] = pn
    # Ensure longest-first ordering helps prefix checks elsewhere if needed
    base_models.sort(key=len, reverse=True)
    return spec_pn_map, base_models


def classify_match(
    raw_pn: str,
    spec_index: Dict[str, str],
    base_models: List[str],
    allow_ctrl_strip: bool = False,
) -> Tuple[str, Optional[str], Optional[str], str]:
    """
    Returns (match_type, matched_spec_pn, matched_base_model, reason)
    match_type in {exact, normalized, stripped, base-only, no-base}
    """
    reason_parts: List[str] = []

    # Step 0: canonicalize for comparison (case insens)
    pn0 = canonical(raw_pn)

    # Step 1: exact against spec (case-insens) and zero-pad normalized key space
    pn_norm = canonical(normalize_zeropad(raw_pn))
    if pn_norm in spec_index:
        return "normalized", spec_index[pn_norm], longest_base_prefix(pn_norm, base_models), "zero-pad and case-insensitive match"

    # Step 2: conservative stripping: parenthetical tails, then -E\d+/-ES\d+
    pn1, stripped_paren = strip_parenthetical(pn0)
    pn2, stripped_enc = strip_encoder_order(pn1)
    if stripped_paren:
        reason_parts.append("removed parenthetical tail")
    if stripped_enc:
        reason_parts.append("stripped -E/-ES suffixes")

    # Optional: last-resort controller/system tags
    pn3 = pn2
    if allow_ctrl_strip:
        for tag in OPTIONAL_CTRL_TAGS:
            if pn3.endswith(tag):
                pn3 = pn3[: -len(tag)]
                reason_parts.append(f"stripped {tag}")
                pn3 = pn3.rstrip("-")
                break

    pn3 = normalize_zeropad(pn3)
    pn3 = canonical(pn3)

    if pn3 in spec_index:
        return "stripped", spec_index[pn3], longest_base_prefix(pn3, base_models), "; ".join(reason_parts) or "stripped to match"

    # Step 3: base-only association if base exists
    base = longest_base_prefix(pn3, base_models)
    if base:
        return "base-only", None, base, "no specific PN under base in specs"

    # Step 4: no base
    return "no-base", None, None, "no base model found in specs"


def iter_results_part_numbers(results: list) -> Iterable[str]:
    for r in results:
        v = r.get("BasePartNum") or r.get("part_number") or r.get("BasePartNumber")
        if v:
            yield str(v)


def write_mapping_csv(rows: Iterable[Dict[str, Optional[str]]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fieldnames = [
        "results_part_number",
        "match_type",
        "matched_spec_part_number",
        "matched_base_model",
        "reason",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare Results BasePartNum to Product Specs part numbers with conservative normalization.")
    ap.add_argument("--specs", required=True, help="Path to Product Specs.json")
    ap.add_argument("--results", required=True, help="Path to Results.json")
    ap.add_argument("--out", default=os.path.join("reports", "part_number_mapping.csv"), help="Output CSV path")
    ap.add_argument("--allow-ctrl-strip", action="store_true", help="Allow stripping trailing -ACS/-ASR as last resort")
    args = ap.parse_args()

    specs = load_json(args.specs)
    results = load_json(args.results)

    spec_index, base_models = build_spec_index(specs)

    stats = {
        "normalized": 0,
        "stripped": 0,
        "base-only": 0,
        "no-base": 0,
        "total_results": 0,
        "unique_results": 0,
    }

    seen = set()
    output_rows: List[Dict[str, Optional[str]]] = []
    for rpn in iter_results_part_numbers(results):
        stats["total_results"] += 1
        if rpn not in seen:
            seen.add(rpn)
            mtype, spec_pn, base, reason = classify_match(rpn, spec_index, base_models, allow_ctrl_strip=args.allow_ctrl_strip)
            stats[mtype] += 1
            output_rows.append({
                "results_part_number": rpn,
                "match_type": mtype,
                "matched_spec_part_number": spec_pn,
                "matched_base_model": base,
                "reason": reason,
            })

    stats["unique_results"] = len(seen)

    write_mapping_csv(output_rows, args.out)

    # Print a concise summary for terminal output
    print(f"Spec bases: {len(base_models)} | Spec PNs: {len(spec_index)}")
    print(f"Unique results PNs: {stats['unique_results']} (from {stats['total_results']} rows)")
    print(f"Matches -> normalized: {stats['normalized']}, stripped: {stats['stripped']}, base-only: {stats['base-only']}, no-base: {stats['no-base']}")
    print(f"Wrote mapping CSV: {args.out}")


if __name__ == "__main__":
    main()
