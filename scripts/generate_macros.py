import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "calories.csv"
OUT = ROOT / "data" / "calories_macros.csv"

def compute_macros(cal):
    # Start with target percentages: protein 15%, fat 35%, carbs remainder
    p = max(1, round((0.15 * cal) / 4))
    f = max(1, round((0.35 * cal) / 9))
    # remaining calories allocated to carbs
    rem = cal - (4 * p + 9 * f)
    c = max(0, round(rem / 4))

    # Adjust to reduce error: try to match calories as closely as possible
    def total(p, c, f):
        return 4 * p + 4 * c + 9 * f

    t = total(p, c, f)
    # Greedy adjustment of carbs, then fat, then protein
    if t != cal:
        diff = cal - t
        # Prefer changing carbs (4 kcal per g)
        adj_c = int(round(diff / 4))
        c += adj_c
        t = total(p, c, f)

    # If still off by a few calories, tweak fat (9 kcal/g) or protein (4 kcal/g)
    while total(p, c, f) < cal:
        # add 1 g carbs if it helps
        c += 1
        if total(p, c, f) >= cal:
            break
        # else add 1 g fat
        f += 1
        if total(p, c, f) >= cal:
            break

    while total(p, c, f) > cal and c > 0:
        c -= 1

    # Ensure non-negative
    p = max(0, p)
    c = max(0, c)
    f = max(0, f)

    return p, c, f


def main():
    if not IN.exists():
        print(f"Input file not found: {IN}")
        return

    rows = []
    with IN.open(newline='', encoding='utf-8') as rf:
        reader = csv.DictReader(rf)
        for r in reader:
            name = r['food']
            try:
                cal = int(float(r['calorie']))
            except Exception:
                cal = 0
            p, c, f = compute_macros(cal)
            rows.append({'food': name, 'calorie': str(cal), 'protein': str(p), 'carbs': str(c), 'fat': str(f)})

    with OUT.open('w', newline='', encoding='utf-8') as wf:
        fieldnames = ['food', 'calorie', 'protein', 'carbs', 'fat']
        writer = csv.DictWriter(wf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {OUT} with {len(rows)} rows")


if __name__ == '__main__':
    main()
