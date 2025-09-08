
#!/usr/bin/env python3
"""
Convert a rational number to its binary representation (base-2) with exact arithmetic.

- Accepts input either as a fraction "p/q", an integer "n", or a decimal string like "0.625".
- Uses Python's fractions.Fraction to avoid floating-point rounding errors.
- Produces a terminating binary expansion when possible (denominator power-of-two after reduction),
  otherwise detects repeating patterns and prints the repeating part in parentheses.
"""
from fractions import Fraction
import sys

def parse_rational(s: str) -> Fraction:
    """Parse input string into a Fraction (supports 'p/q' or decimal/integer strings)."""
    s = s.strip()
    if '/' in s:
        p_str, q_str = s.split('/', 1)
        return Fraction(int(p_str), int(q_str))
    # Fraction can parse decimal strings exactly as rationals
    return Fraction(s)

def integer_to_binary(n: int) -> str:
    """Convert a non-negative integer to a binary string without '0b' prefix."""
    if n == 0:
        return "0"
    bits = []
    while n > 0:
        bits.append(str(n & 1))  # append least significant bit
        n >>= 1                  # shift right by 1 (divide by 2)
    return ''.join(reversed(bits))

def rational_to_binary(frac: Fraction, max_frac_bits: int = 256) -> str:
    """
    Convert a Fraction to binary string. If fractional part repeats, put the repeating
    cycle in parentheses, e.g., 1/3 -> '0.(01)' because 0.010101...
    max_frac_bits is a safety cap to avoid infinite loops on very long repeats.
    """
    if frac == 0:
        return "0"

    # Handle sign first
    sign = '-' if frac < 0 else ''
    frac = abs(frac)

    # Integer part
    integer_part = frac.numerator // frac.denominator
    int_bits = integer_to_binary(integer_part)

    # Remainder determines the fractional bits
    remainder = frac.numerator % frac.denominator
    if remainder == 0:
        # It is an integer; no fractional part
        return f"{sign}{int_bits}"

    # Fractional part generation with cycle detection
    denom = frac.denominator
    seen = {}            # remainder -> index in the bits list (to detect repetition)
    bits = []

    idx = 0
    repeating_start = None

    while remainder and idx < max_frac_bits:
        if remainder in seen:
            # We found a repeating cycle
            repeating_start = seen[remainder]
            break
        seen[remainder] = idx
        remainder *= 2
        bit = remainder // denom
        bits.append(str(bit))
        remainder = remainder % denom
        idx += 1

    if remainder == 0:
        # Terminating fractional part
        frac_str = ''.join(bits)
        return f"{sign}{int_bits}.{frac_str}"
    else:
        # Repeating fractional part: put cycle in parentheses
        non_rep = ''.join(bits[:repeating_start])
        rep = ''.join(bits[repeating_start:])
        return f"{sign}{int_bits}.{non_rep}({rep})"

def demo():
    examples = [
        "1621",          # integer case
        "443/2048",      # denominator is power of two (terminates)
        "1/3",           # repeating
        "5/8",           # 0.101
        "-7/10",         # negative, repeating
        "0.625",         # decimal exact: 0.101
    ]
    for s in examples:
        f = parse_rational(s)
        print(f"{s:<10} -> {rational_to_binary(f)}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments: run demo
        demo()
    else:
        # Convert each argument and print the binary expansion
        for arg in sys.argv[1:]:
            f = parse_rational(arg)
            print(rational_to_binary(f))
