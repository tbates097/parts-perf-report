#!/usr/bin/env python3
import argparse
import json
from typing import List


def load_specs(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def mech_keys(specs, pn: str):
    if not isinstance(specs, dict):
        return None
    for base, entries in specs.items():
        if isinstance(entries, list):
            for item in entries:
                if isinstance(item, dict) and pn in item:
                    mech = item[pn].get('specifications', {}).get('mechanical', {})
                    return list(mech.keys())
        elif isinstance(entries, dict):
            if pn in entries:
                mech = entries[pn].get('specifications', {}).get('mechanical', {})
                return list(mech.keys())
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--specs', required=True)
    ap.add_argument('pns', nargs='+')
    args = ap.parse_args()

    specs = load_specs(args.specs)
    for pn in args.pns:
        keys = mech_keys(specs, pn)
        print(f"PN {pn} ->")
        if keys is None:
            print("  not found")
        else:
            for k in keys:
                print("  ", k)


if __name__ == '__main__':
    main()
