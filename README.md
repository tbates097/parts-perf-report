# Parts Performance vs Specs Reporter

This project provides a Python script to compare measured test results for parts against the corresponding specifications and produce a comprehensive report.

## Inputs
- tests JSON: part numbers and measurement results per unit.
- specs JSON: specifications per part number (min/max or equals for each metric).

## Sample formats
See `data/tests.sample.json` and `data/specs.sample.json`.

## Usage
Activate your preferred environment (you prefer the `tbates` conda env for running scripts).

```powershell
# Example
python .\report.py --tests .\data\tests.sample.json --specs .\data\specs.sample.json --out .\reports\report.csv
```

The output is a CSV with per-metric pass/fail and margins. Create the `reports` directory if needed.

## Notes
- Specs can contain `min`, `max`, and/or `equals` constraints per metric.
- Unknown metrics are included in the report and marked accordingly.
