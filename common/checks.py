"""Answer checking for exercises, especially the ✍️ pen-and-paper ones.

    answer = None          # TODO: your hand-computed result
    check(answer, A @ b)   # ✅ if it matches, otherwise a helpful error

Works for numbers, lists/arrays (compared element-wise with a tolerance), booleans/strings (exact),
and SymPy expressions (compared symbolically, so `2*x + 2` matches `2*(x + 1)`).
"""
import numpy as np


def _is_sympy(obj):
    return type(obj).__module__.startswith("sympy")


def check(answer, expected, tol=1e-6, name="answer"):
    if answer is None:
        raise AssertionError(f"⏳ {name} is still None. Replace it with your result first.")

    if _is_sympy(answer) or _is_sympy(expected):
        import sympy as sp
        ok = sp.simplify(sp.sympify(answer) - sp.sympify(expected)) == 0
        if not ok:
            raise AssertionError(f"❌ {name} = {answer} does not simplify to the expected expression.")
    elif isinstance(expected, (bool, str, np.bool_)):
        if answer != expected:
            raise AssertionError(f"❌ {name} = {answer!r}, which is not right. Re-read the question.")
    else:
        a = np.asarray(answer, dtype=float)
        e = np.asarray(expected, dtype=float)
        if a.shape != e.shape:
            raise AssertionError(f"❌ {name} has shape {a.shape}, expected shape {e.shape}.")
        if not np.allclose(a, e, atol=tol, rtol=tol):
            raise AssertionError(f"❌ {name} = {np.round(a, 4).tolist()} is not right. Check each step.")
    print(f"✅ {name} is correct")
