"""
Score rally detection against your hand labels.

Usage:
    python score_rallies.py results/match2/rallies.json results/match2/labels.csv
    python score_rallies.py <rallies.json> <labels.csv> --tolerance 1.5

A detected rally "matches" a real one when they overlap and both the start and
end are within --tolerance seconds... or, more leniently, when they overlap at
all (reported separately, so you can see whether the problem is finding rallies
or getting their boundaries exactly right).
"""

import argparse
import csv
import json


def fmt(seconds):
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:04.1f}"


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def match(detected, truth):
    """Greedy one-to-one matching by largest overlap."""
    pairs = []
    for i, d in enumerate(detected):
        for j, t in enumerate(truth):
            ov = overlap(d, t)
            if ov > 0:
                pairs.append((ov, i, j))
    pairs.sort(reverse=True)
    used_d, used_t, matches = set(), set(), []
    for ov, i, j in pairs:
        if i not in used_d and j not in used_t:
            used_d.add(i); used_t.add(j); matches.append((i, j))
    return matches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rallies_json")
    p.add_argument("labels_csv")
    p.add_argument("--tolerance", type=float, default=1.0,
                   help="seconds a start/end can be off and still count as correct")
    args = p.parse_args()

    with open(args.rallies_json) as fh:
        detected = [(r["start"], r["end"]) for r in json.load(fh)["rallies"]]
    with open(args.labels_csv) as fh:
        truth = [(float(r["start"]), float(r["end"])) for r in csv.DictReader(fh) if r.get("end")]
    detected.sort(); truth.sort()

    matches = match(detected, truth)
    m_d = {i for i, _ in matches}
    m_t = {j for _, j in matches}

    start_err = [abs(detected[i][0] - truth[j][0]) for i, j in matches]
    end_err = [abs(detected[i][1] - truth[j][1]) for i, j in matches]
    exact = sum(1 for s, e in zip(start_err, end_err) if s <= args.tolerance and e <= args.tolerance)

    print(f"Real rallies: {len(truth)}   Detected: {len(detected)}\n")
    print("  real                    detected                start err  end err")
    rows = [(truth[j][0], "match", i, j) for i, j in matches]
    rows += [(truth[j][0], "missed", None, j) for j in range(len(truth)) if j not in m_t]
    rows += [(detected[i][0], "fake", i, None) for i in range(len(detected)) if i not in m_d]
    for _, kind, i, j in sorted(rows):
        real = f"{fmt(truth[j][0])} - {fmt(truth[j][1])}" if j is not None else "-"
        det = f"{fmt(detected[i][0])} - {fmt(detected[i][1])}" if i is not None else "-"
        if kind == "match":
            se = detected[i][0] - truth[j][0]
            ee = detected[i][1] - truth[j][1]
            flag = "" if abs(se) <= args.tolerance and abs(ee) <= args.tolerance else "  <- off"
            print(f"  {real:<22}  {det:<22}  {se:+7.1f}s  {ee:+7.1f}s{flag}")
        elif kind == "missed":
            print(f"  {real:<22}  {'MISSED':<22}")
        else:
            print(f"  {'-':<22}  {det:<22}  FAKE (no real rally here)")

    n = max(len(truth), 1)
    print("\nSummary")
    print(f"  Found (overlapping a real rally):   {len(matches)}/{len(truth)}  ({100 * len(matches) / n:.0f}%)")
    print(f"  Found with start AND end within {args.tolerance:g}s: {exact}/{len(truth)}  ({100 * exact / n:.0f}%)")
    print(f"  Missed rallies:  {len(truth) - len(matches)}")
    print(f"  Fake rallies:    {len(detected) - len(matches)}")
    if matches:
        print(f"  Average start error: {sum(start_err) / len(start_err):.1f}s   "
              f"average end error: {sum(end_err) / len(end_err):.1f}s")
        late_ends = sum(1 for i, j in matches if detected[i][1] - truth[j][1] > args.tolerance)
        if late_ends:
            print(f"  {late_ends} rallies end too LATE (e.g. shuttle pickup counted as play)")


if __name__ == "__main__":
    main()