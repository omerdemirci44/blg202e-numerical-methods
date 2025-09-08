
#!/usr/bin/env python3
"""
BLG202E - HW3 Q3(b)
Program implementing the Lagrange interpolation method for a quadratic through 3 points.
Demonstrates evaluation at x = 0 for the dataset in Q3.
"""
from typing import List
import math

def lagrange_eval(x: float, xs: List[float], ys: List[float]) -> float:
    """
    Evaluate the Lagrange interpolation polynomial passing through (xs[i], ys[i]).
    This implementation is O(n^2), suitable for small n (e.g., n=3 here).
    """
    n = len(xs)
    total = 0.0
    for i in range(n):
        # Compute L_i(x)
        Li = 1.0
        xi = xs[i]
        for j in range(n):
            if j == i:
                continue
            denom = xi - xs[j]
            if denom == 0:
                raise ZeroDivisionError("Duplicate x values in data points.")
            Li *= (x - xs[j]) / denom
        total += ys[i] * Li
    return total

def demo():
    xs = [-1.2, 0.3, 1.1]
    ys = [-5.76, -5.61, -3.69]
    xq = 0.0
    val = lagrange_eval(xq, xs, ys)
    print(f"P({xq}) = {val:.12f}")

if __name__ == "__main__":
    demo()
