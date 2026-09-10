"""The two statistics this project actually leans on, written out rather than imported.

Both are three lines of arithmetic and both get quoted in the report, so they are here
in a form that can be read and defended line by line. `--check` verifies them against
scipy/sklearn so that "I wrote it myself" does not mean "I got it wrong".
"""

from __future__ import annotations

import math
from collections import Counter


def cohen_kappa(a: list, b: list) -> float:
    """Agreement between two raters, corrected for agreement by luck.

        kappa = (p_o - p_e) / (1 - p_e)

    p_o is how often they actually agreed. p_e is how often they would have agreed if
    both had just guessed at random using their own observed class frequencies.

    Raw agreement alone is worthless as evidence: two raters who both always answer the
    most common class agree ~90% of the time while knowing nothing. Kappa subtracts
    exactly that.

    Reading: <0.20 slight, 0.21-0.40 fair, 0.41-0.60 moderate, 0.61-0.80 substantial,
    0.81+ almost perfect. Negative means worse than guessing.

    The trap (the "kappa paradox"): when one class dominates, p_e is already high, so
    kappa can look poor even at 90% raw agreement. Always report both numbers together.
    """
    if len(a) != len(b):
        raise ValueError("rater lists must be the same length")
    n = len(a)
    if n == 0:
        return float("nan")

    p_o = sum(1 for x, y in zip(a, b) if x == y) / n

    ca, cb = Counter(a), Counter(b)
    p_e = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))

    if p_e == 1.0:
        # Both raters used exactly one class, the same one. They agree completely and
        # chance also predicts complete agreement, so kappa is undefined, not perfect.
        return float("nan")
    return (p_o - p_e) / (1 - p_e)


def raw_agreement(a: list, b: list) -> float:
    return sum(1 for x, y in zip(a, b) if x == y) / len(a) if a else float("nan")


# The exact two-sided 95% normal quantile. Textbooks round it to 1.96; that rounding
# moves an interval endpoint in the 6th decimal place, which is invisible in a report
# but makes a cross-check against statsmodels fail for no real reason.
Z95 = 1.959963984540054


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float, float]:
    """95% confidence interval for a proportion. Returns (point, low, high).

    Wilson rather than the textbook p +- z*sqrt(p(1-p)/n) because that one misbehaves
    badly near 0 and 1 -- at 100% correct it reports an interval of zero width, which
    would let a small eval claim certainty it has not earned.

    This is why the results table has a +- column. At n=140 an 85% score carries roughly
    +-6 points, so a 3-point gap between two configs is not a result.
    """
    if n == 0:
        return (float("nan"),) * 3
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def fmt_ci(successes: int, n: int) -> str:
    """`85% +-6` -- point estimate and half-width, for the results table."""
    if n == 0:
        return "  n/a"
    p, lo, hi = wilson(successes, n)
    return f"{p:>4.0%} +-{(hi - lo) / 2 * 100:>2.0f}"


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        ok = True
        try:
            from sklearn.metrics import cohen_kappa_score
            cases = [
                (["a", "b", "a", "c", "b", "a"], ["a", "b", "c", "c", "b", "a"]),
                (["x"] * 9 + ["y"], ["x"] * 8 + ["y", "y"]),
                (list("abcabcabc"), list("abcabcabc")),
            ]
            for a, b in cases:
                mine, theirs = cohen_kappa(a, b), cohen_kappa_score(a, b)
                match = abs(mine - theirs) < 1e-9
                ok &= match
                print(f"  kappa mine={mine:+.6f} sklearn={theirs:+.6f} "
                      f"{'OK' if match else 'MISMATCH'}")
        except ImportError:
            print("  sklearn not installed, skipping kappa cross-check")
        try:
            from statsmodels.stats.proportion import proportion_confint
            for s, n in [(119, 140), (140, 140), (0, 140), (70, 140)]:
                _, lo, hi = wilson(s, n)
                tlo, thi = proportion_confint(s, n, method="wilson")
                match = abs(lo - tlo) < 1e-9 and abs(hi - thi) < 1e-9
                ok &= match
                print(f"  wilson {s}/{n} mine=({lo:.6f},{hi:.6f}) "
                      f"statsmodels=({tlo:.6f},{thi:.6f}) {'OK' if match else 'MISMATCH'}")
        except ImportError:
            print("  statsmodels not installed, skipping wilson cross-check")
        sys.exit(0 if ok else 1)

    print("  n=140, 85% correct ->", fmt_ci(119, 140))
    print("  n=140, 82% correct ->", fmt_ci(115, 140))
    print("  the intervals overlap, so that 3-point gap is not a result")
