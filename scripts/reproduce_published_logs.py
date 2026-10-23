#!/usr/bin/env python3
"""Reproduce the published intervals and sign tests using only the standard library.

Run: python3 scripts/reproduce_published_logs.py --check
CSV geometry is transcribed from the config snapshots cited in the document;
its hashes identify those snapshots. No network, GPU or private data is needed.
"""
import argparse
import csv
from decimal import Decimal
from fractions import Fraction
import json
from math import comb
from pathlib import Path
import re
from statistics import median

ROOT = Path(__file__).resolve().parents[1]


def memory_bounds(gib, rule):
    target = Decimal(gib)
    if not target.is_finite() or target <= 0 or rule not in ('printf_2f', 'round_2_str'):
        raise ValueError('positive finite GiB and known formatter required')

    def display(pool):
        value = pool / 2**30
        return Decimal(('%.2f' % value) if rule == 'printf_2f' else str(round(value, 2)))

    def first(strict):
        low, high = 0, int((target + 1) * 2**30)
        while low < high:
            mid = (low + high) // 2
            if display(mid) > target or (not strict and display(mid) == target):
                high = mid
            else:
                low = mid + 1
        return low

    low, high = first(False), first(True) - 1
    if low > high or display(low) != target or display(high) != target:
        raise ValueError('GiB value has no formatter preimage')
    return low, high


def sign_test(values):
    negative, zero, positive = (sum(v < 0 for v in values), sum(v == 0 for v in values), sum(v > 0 for v in values))
    n = negative + positive
    p = min(Fraction(1), Fraction(2 * sum(comb(n, j) for j in range(min(negative, positive) + 1)), 2**n))
    return [str(negative), str(zero), str(positive), f'{p} = {float(p)}']


def reproduce(path):
    intervals, unique, issues = [], {}, {}
    with path.open(newline='', encoding='utf-8') as stream:
        records = list(csv.DictReader(stream))
    if not records or len({r['record'] for r in records}) != len(records):
        raise ValueError('nonempty records with unique identifiers required')
    for r in records:
        layers, heads, dim, block, length, tp = [int(r[k]) for k in ('layers', 'kv_heads', 'head_dim', 'block', 'max_length', 'tp')]
        if min(layers, heads, dim, block, length, tp) <= 0 or length % block or (heads >= tp and heads % tp):
            raise ValueError('positive uniform geometry and block-aligned length required')
        dtype = r['runtime_dtype'] if r['kv_dtype'] == 'auto' else r['kv_dtype']
        width = {'float16': 2, 'bfloat16': 2, 'fp8': 1, 'fp8_e4m3': 1}[dtype]
        stride = layers * block * max(1, heads // tp) * 2 * dim * width
        low, high = memory_bounds(r['printed_gib'], r['formatter'])

        def capacity(pool):
            blocks = pool // stride
            concurrency = blocks / (length // block)
            # All admitted lengths are block-aligned; retain source float evaluation.
            tokens = int(concurrency * length) if r['report_rule'] == 'concurrency_tokens' else blocks * block
            return tokens, f'{concurrency:.2f}'

        if r['report_rule'] not in ('concurrency_tokens', 'block_tokens'):
            raise ValueError('unknown source report rule')
        lo, hi, centre = capacity(low), capacity(high), capacity((low + high) // 2)
        delta = centre[0] - int(r['reported_tokens'])
        key = tuple(r[k] for k in ('version', 'config_sha256', 'runtime_dtype', 'kv_dtype', 'backend', 'block', 'max_length', 'tp', 'printed_gib', 'reported_tokens', 'reported_concurrency'))
        duplicate = unique[key][0] if key in unique else 'none'
        if duplicate == 'none':
            unique[key] = (r['record'], delta)
            issues.setdefault(r['issue_url'], []).append(delta)
        intervals.append([r['record'], r['printed_gib'], f"{int(r['reported_tokens']):,}",
                          f'[{lo[0]:,}, {hi[0]:,}]', f'{delta:+d}', r['reported_concurrency'],
                          f'[{lo[1]}, {hi[1]}]', duplicate])
    means = [Fraction(sum(v), len(v)) for v in issues.values()]
    medians = [median(v) for v in issues.values()]
    return {'intervals': intervals, 'counts': [len(records), len(unique), len(issues)],
            'signs': [sign_test([v[1] for v in unique.values()]), sign_test(means), sign_test(medians)],
            'issues': [[url.rsplit('/', 1)[1], str(sum(r['issue_url'] == url for r in records)), str(len(v)),
                        str(Fraction(sum(v), len(v))), str(median(v))] for url, v in issues.items()]}


def check(result, document):
    def table(section):
        part = document.split('## ' + section + '\n', 1)[1].split('\n## ', 1)[0]
        rows = [[re.sub(r'\[([^]]+)\]\([^)]+\)', r'\1', cell.strip()) for cell in line.split('|')[1:-1]]
                for line in part.splitlines() if line.startswith('|')]
        return rows[2:]
    if result['intervals'] != table('Interval table'):
        raise ValueError('published interval table differs')
    stats = table('Residual signs and independence')
    if result['signs'] != [r[1:] for r in stats[:3]] or result['issues'] != stats[5:]:
        raise ValueError('published sign or issue table differs')
    counts = re.search(r'\*\*(\d+) records, (\d+) unique input-observation tuples,\s*and (\d+) issue clusters\*\*', document)
    if not counts or list(map(int, counts.groups())) != result['counts']:
        raise ValueError('published counts differ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=ROOT / 'docs/published-log-inputs.csv')
    parser.add_argument('--document', type=Path, default=ROOT / 'docs/published-log-validation.md')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = reproduce(args.csv)
    if args.check:
        check(result, args.document.read_text(encoding='utf-8'))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
