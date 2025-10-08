`
# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.
``

Project overview
- Purpose: Compare measured part performance (tests JSON) to specifications (specs JSON) and produce a CSV report with pass/fail and margins.
- Primary entrypoint: report.py (CLI)
- Data samples: data/tests.sample.json, data/specs.sample.json

Environment
- Windows + PowerShell (pwsh)
- Preferred environment: conda environment named tbates
  - Activate before running any commands:
    - pwsh: conda activate tbates

Common commands
- Run script (real data):
  - python .\report.py --tests .\path\to\tests.json --specs .\path\to\specs.json --out .\reports\report.csv
  - Note: The reports directory is created automatically if it doesn’t exist.
- Run script with samples (from README):
  - python .\report.py --tests .\data\tests.sample.json --specs .\data\specs.sample.json --out .\reports\report.csv
- Build: N/A (single Python script; no packaging config present)
- Lint/format: N/A (no tool config files present)
- Tests: N/A (no tests present)
  - Running a single test is not applicable until a test suite exists.

High-level architecture and data flow
- report.py provides a CLI that orchestrates the entire workflow:
  1) load_json(path) — loads tests/specs JSON files.
  2) normalize_specs(data) — accepts either of these shapes and returns a mapping of part_number -> metric -> {min|max|equals}:
     - { "parts": { PN: { metric: {min,max,equals} } } } or { PN: { metric: {min,max,equals} } }
     - Validates constraints exist per metric (at least one of min, max, equals).
  3) normalize_tests(data) — accepts either a list of entries or a mapping and returns a list of entries:
     - Accepts: [ {part_number, unit_id?, results?}, ... ]
       or { PN: [ {unit_id?, results?} ... ] } or { PN: { ...flattened numeric metrics... } }
     - Ensures each entry has: part_number, optional unit_id, and results {metric: numeric_value}.
     - Performs a light “flatten” if results isn’t provided but numeric metrics are top-level.
     - Skips invalid entries.
  4) evaluate(value, spec) — computes pass/fail per metric:
     - If equals is specified, requires exact match.
     - Otherwise checks min/max; returns booleans and deltas to min/max.
  5) generate_rows(tests, specs) — yields row dicts for CSV:
     - For unknown part numbers or metrics, emits rows with note fields (e.g., “No specs for part_number”, “No spec for metric”).
     - For non-numeric values in tests, emits a row marked as pass=False with note="Non-numeric value".
  6) write_csv(rows, out_path) — writes all rows with header to the specified CSV file, creating parent directories as needed.

Key behaviors and conventions
- Specs constraints per metric can include any of: min, max, equals. At least one is required.
- Unknown metrics and unknown part numbers are still reported with explanatory notes.
- CSV schema columns (in order):
  - part_number, unit_id, metric, value, spec_min, spec_max, pass, delta_to_min, delta_to_max, note

Files of interest
- report.py — All logic and CLI.
- README.md — Quick usage example and data format notes.
- data/ — Sample inputs (tests.sample.json, specs.sample.json) and example large JSONs.

Additional notes for agents
- Prefer running within the tbates conda environment (per user rule).
- Commands shown assume pwsh on Windows; adjust path separators if using a different shell.
