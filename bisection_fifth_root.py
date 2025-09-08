
#!/usr/bin/env python3
"""
Bisection method for computing the real fifth root: x ≈ a^(1/5) with |x - a^(1/5)| ≤ eps.

- f(x) = x^5 - a is strictly increasing on R, so there is a unique real root for any real a.
- We choose an interval [lo, hi] that brackets the root (f(lo) ≤ 0 ≤ f(hi)).
- We bisect until the half-interval length is ≤ eps, ensuring |x* - mid| ≤ eps.
"""
import sys
import math

def fifth_root_bisect(a: float, eps: float, max_iter: int = 2000):
    """
    Return x such that |x - a^(1/5)| ≤ eps using bisection.
    The guarantee is achieved by stopping when (hi - lo)/2 ≤ eps with [lo, hi] bracketing the root.
    """
    # Trivial case
    if a == 0:
        return 0.0, 0

    # Choose a bracketing interval depending on the sign of a
    f = lambda x: x**5 - a

    if a > 0:
        lo, hi = 0.0, max(1.0, a)  # f(lo) = -a ≤ 0, f(hi) ≥ 0
    else:
        lo, hi = min(a, -1.0), 0.0 # f(lo) ≤ 0, f(hi) = -a ≥ 0

    # Bisection loop
    it = 0
    while it < max_iter and (hi - lo) / 2.0 > eps:
        mid = (lo + hi) / 2.0
        if f(mid) >= 0:
            hi = mid   # root lies in [lo, mid]
        else:
            lo = mid   # root lies in [mid, hi]
        it += 1

    x = (lo + hi) / 2.0
    return x, it

def demo():
    tests = [
        (7.0, 1e-6),
        (0.03125, 1e-8),   # 1/32, exact fifth root is 1/2 = 0.5
        (1.0, 1e-12),
        (-32.0, 1e-7),     # negative a works (real fifth root is -2)
    ]
    for a, eps in tests:
        x, it = fifth_root_bisect(a, eps)
        # For verification / info only (not used by the method itself)
        true_val = math.copysign(abs(a)**(1/5), a)
        err = abs(x - true_val)
        print(f"a={a:<10} eps={eps:<10} -> x={x:.12f}  |x - a^(1/5)|={err:.3e}  iters={it}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        demo()
    else:
        if len(sys.argv) != 3:
            print("Usage: python bisection_fifth_root.py <a> <eps>")
            sys.exit(1)
        a = float(sys.argv[1])
        eps = float(sys.argv[2])
        x, it = fifth_root_bisect(a, eps)
        print(f"{x}")
