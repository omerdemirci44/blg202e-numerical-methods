
#!/usr/bin/env python3
"""
BLG202E - HW2 Q1
Newton and Secant methods for f(x) = 4 ln(x) - x, where f'(x) = 4/x - 1.

- Computes the first N iterations and prints a table with: n, x_n, f(x_n), f'(x_n), |x_n - x_{n-1}|.
- Includes functions for error estimation and empirical convergence-order estimation.
- Notes:
    * Domain requires x > 0 (because of ln).
    * For Newton: a reasonable x0 > 0 is required.
    * For Secant: need two initial guesses x0, x1 > 0.
"""
import math
from typing import Callable, List, Tuple, Optional

# Define the function and its derivative
def f(x: float) -> float:
    return 4.0 * math.log(x) - x

def df(x: float) -> float:
    return 4.0 / x - 1.0

def newton(f: Callable[[float], float],
           df: Callable[[float], float],
           x0: float,
           max_iter: int = 50,
           tol: float = 1e-12) -> List[Tuple[int, float, float, float, Optional[float]]]:
    """
    Newton's method. Returns a list of tuples (n, x_n, f(x_n), f'(x_n), |x_n - x_{n-1}|).
    Stops when the increment <= tol or max_iter is reached.
    """
    if x0 <= 0:
        raise ValueError("Newton requires x0 > 0 due to ln(x).")
    table = []
    x_prev = None
    x = x0
    for n in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        if dfx == 0.0:
            raise ZeroDivisionError("Derivative became zero during Newton iterations.")
        x_new = x - fx / dfx
        inc = None if x_prev is None else abs(x_new - x)
        table.append((n, x_new, f(x_new), df(x_new), inc))
        if inc is not None and inc <= tol:
            break
        x_prev, x = x, x_new
        if x <= 0:
            raise ValueError("Iterate left the domain x>0; choose a different initial guess.")
    return table

def secant(f: Callable[[float], float],
           x0: float,
           x1: float,
           max_iter: int = 50,
           tol: float = 1e-12) -> List[Tuple[int, float, float, Optional[float]]]:
    """
    Secant method. Returns a list of tuples (n, x_n, f(x_n), |x_n - x_{n-1}|).
    Stops when increment <= tol or max_iter is reached.
    """
    if x0 <= 0 or x1 <= 0:
        raise ValueError("Secant requires x0, x1 > 0 due to ln(x).")
    table = []
    x_prev = x0
    x = x1
    f_prev = f(x_prev)
    for n in range(1, max_iter + 1):
        fx = f(x)
        denom = (fx - f_prev)
        if denom == 0.0:
            raise ZeroDivisionError("Zero denominator in Secant iteration.")
        x_new = x - fx * (x - x_prev) / denom
        inc = abs(x_new - x)
        table.append((n, x_new, f(x_new), inc))
        if inc <= tol:
            break
        x_prev, f_prev, x = x, fx, x_new
        if x <= 0:
            raise ValueError("Iterate left the domain x>0; choose different initial guesses.")
    return table

def absolute_increments(xs: List[float]) -> List[float]:
    """Return sequence of absolute increments |x_n - x_{n-1}| (first element is None-equivalent -> skipped)."""
    incs = []
    for i in range(1, len(xs)):
        incs.append(abs(xs[i] - xs[i-1]))
    return incs

def empirical_order(xs: List[float]) -> Optional[float]:
    """
    Estimate empirical convergence order p using three consecutive errors relative to the last iterate x*:
        e_n   = |x_n - x*|
        p_hat = ln(e_{n+1}/e_n) / ln(e_n/e_{n-1})
    Uses the last 4 iterates if available. Returns None if not enough data.
    """
    if len(xs) < 4:
        return None
    x_star = xs[-1]  # best available proxy for the true root
    es = [abs(x - x_star) for x in xs[-4:]]  # e_{k-3}, e_{k-2}, e_{k-1}, e_k
    e1, e2, e3, e4 = es
    # Need e_{n-1}, e_n, e_{n+1}; use (e1, e2, e3) and (e2, e3, e4), average if both valid
    vals = []
    for (a, b, c) in [(e1, e2, e3), (e2, e3, e4)]:
        if a > 0 and b > 0 and c > 0 and (b != a) and (c != b):
            vals.append(math.log(c/b)/math.log(b/a))
    if not vals:
        return None
    return sum(vals)/len(vals)

def run_demo():
    # Problem data from HW2 Q1:
    # Newton: x0 = 1
    # Secant: x0 = 1, x1 = 2
    newton_tbl = newton(f, df, x0=1.0, max_iter=6, tol=0.0)
    secant_tbl = secant(f, x0=1.0, x1=2.0, max_iter=6, tol=0.0)

    print("=== Newton (first 6 iterations) ===")
    xs_newton = []
    for n, xn, fxn, dfxn, inc in newton_tbl:
        xs_newton.append(xn)
        print(f"{n:2d}  x_n={xn:.12f}  f(x_n)={fxn:.3e}  f'(x_n)={dfxn:.6f}  |Δ|={('—' if inc is None else f'{inc:.3e}')}")
    pN = empirical_order(xs_newton)
    if pN is not None:
        print(f"Empirical convergence order (Newton) ≈ {pN:.3f}")

    print("\n=== Secant (first 6 iterations) ===")
    xs_secant = []
    for n, xn, fxn, inc in secant_tbl:
        xs_secant.append(xn)
        print(f"{n:2d}  x_n={xn:.12f}  f(x_n)={fxn:.3e}  |Δ|={inc:.3e}")
    pS = empirical_order(xs_secant)
    if pS is not None:
        print(f"Empirical convergence order (Secant) ≈ {pS:.3f}")

if __name__ == "__main__":
    # Default: run the demo specified by the homework
    run_demo()
