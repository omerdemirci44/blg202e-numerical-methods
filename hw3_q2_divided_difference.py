
#!/usr/bin/env python3
"""
BLG202E - HW3 Q2(b)
Construct a program implementing the divided difference table and Newton-form
interpolation. Also demonstrates evaluation at x = 4.2 for the dataset in Q2.
"""
from typing import List, Tuple
import math

def divided_differences(xs: List[float], ys: List[float]) -> List[List[float]]:
    """
    Build the (upper-triangular) divided differences table.
    Returns a 2D list 'table' where table[i][j] = f[x_i, ..., x_{i+j}].
    The first column table[i][0] are the y values.
    """
    n = len(xs)
    table = [[0.0]*n for _ in range(n)]
    for i in range(n):
        table[i][0] = ys[i]
    for j in range(1, n):
        for i in range(n - j):
            denom = xs[i + j] - xs[i]
            if denom == 0:
                raise ZeroDivisionError("Duplicate x values encountered in divided differences.")
            table[i][j] = (table[i+1][j-1] - table[i][j-1]) / denom
    return table

def newton_coeffs(table: List[List[float]]) -> List[float]:
    """Return coefficients a0, a1, ..., a_{n-1} from the divided differences table (the first row)."""
    return [table[0][j] for j in range(len(table))]

def newton_eval(x: float, xs: List[float], coeffs: List[float]) -> float:
    """Evaluate Newton-form polynomial at x using Horner-like nested multiplication."""
    n = len(coeffs)
    result = coeffs[-1]
    for k in range(n-2, -1, -1):
        result = result * (x - xs[k]) + coeffs[k]
    return result

def print_table(xs: List[float], table: List[List[float]]):
    """Pretty-print the divided differences table with x values."""
    n = len(xs)
    header = ["x"] + [f"f[{','.join(['x'+str(i) for i in range(col+1)])}]" for col in range(n)]
    print("Divided Differences Table (Newton)")
    print("".join(h.center(18) for h in header))
    for i in range(n):
        row = [f"{xs[i]:.6g}"] + [f"{table[i][j]:.6g}" if j < n - i else "" for j in range(n)]
        print("".join(c.center(18) for c in row))

def demo():
    xs = [0, 1, 2, 4, 6]
    ys = [1, 9, 23, 93, 259]
    table = divided_differences(xs, ys)
    print_table(xs, table)
    coeffs = newton_coeffs(table)
    xq = 4.2
    val = newton_eval(xq, xs, coeffs)
    print(f"\nP({xq}) = {val:.12f}")

if __name__ == "__main__":
    demo()
