"""Plotting helpers shared across modules. Notebooks import them as `mplt` (see the setup cell):

    mplt.plot_function(np.sin, -3, 3)
    mplt.plot_vectors([[2, 1], [-1, 2]], labels=["a", "b"])
    mplt.plot_linear_transform(np.array([[1, 1], [0, 1]]))
"""
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]


def _ax(ax, figsize=(5, 5)):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def plot_function(fns, x_min=-5, x_max=5, labels=None, ax=None, n=400, ylim=None, title=None):
    """Plot one function or a list of functions of a single variable."""
    ax = _ax(ax, (6, 4))
    fns = fns if isinstance(fns, (list, tuple)) else [fns]
    x = np.linspace(x_min, x_max, n)
    for i, f in enumerate(fns):
        label = labels[i] if labels else getattr(f, "__name__", None)
        ax.plot(x, f(x), label=label, color=COLORS[i % len(COLORS)])
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    if ylim:
        ax.set_ylim(*ylim)
    if labels or len(fns) > 1:
        ax.legend()
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def plot_vectors(vectors, labels=None, origin=(0, 0), ax=None, lim=None, title=None, colors=None):
    """Draw 2-D vectors as arrows from `origin` (one origin for all, or one per vector)."""
    ax = _ax(ax)
    V = np.atleast_2d(np.asarray(vectors, dtype=float))
    O = np.broadcast_to(np.asarray(origin, dtype=float), V.shape)
    for i, (o, v) in enumerate(zip(O, V)):
        c = (colors or COLORS)[i % len(colors or COLORS)]
        ax.quiver(*o, *v, angles="xy", scale_units="xy", scale=1, color=c, width=0.012)
        if labels:
            ax.annotate(labels[i], o + v, textcoords="offset points", xytext=(5, 5), color=c, fontsize=12)
    pts = np.vstack([O, O + V, [[0, 0]]])
    m = lim or (np.abs(pts).max() + 1)
    ax.set_xlim(-m, m)
    ax.set_ylim(-m, m)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)
    return ax


def plot_linear_transform(A, ax=None, lim=4, title=None, show_unit_square=True):
    """Show how the 2×2 matrix A moves the grid (gray → blue) and the basis vectors e1, e2 (→ red, green)."""
    A = np.asarray(A, dtype=float)
    ax = _ax(ax)
    span = 3 * lim                          # draw a wider grid than the view so stretched lines still fill it
    ts = np.arange(-span, span + 1)
    line = np.linspace(-span, span, 200)
    for t in ts:
        for pts in (np.stack([line, np.full_like(line, t)]), np.stack([np.full_like(line, t), line])):
            ax.plot(*pts, color="lightgray", lw=0.6)
            ax.plot(*(A @ pts), color=COLORS[0], lw=0.6, alpha=0.6)
    if show_unit_square:
        sq = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]], dtype=float)
        ax.fill(*(A @ sq), color=COLORS[0], alpha=0.15)
    for v, c, name in ((A[:, 0], COLORS[1], "A e1"), (A[:, 1], COLORS[2], "A e2")):
        ax.quiver(0, 0, *v, angles="xy", scale_units="xy", scale=1, color=c, width=0.012)
        ax.annotate(name, v, textcoords="offset points", xytext=(5, 5), color=c)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(title or f"A = {A.round(2).tolist()}")
    return ax


def plot_contour_with_path(f, path=None, xlim=(-3, 3), ylim=(-3, 3), ax=None, levels=30, title=None):
    """Contour plot of f(x, y) (vectorized), optionally with an optimization path of shape (steps, 2)."""
    ax = _ax(ax, (6, 5))
    xs, ys = np.linspace(*xlim, 200), np.linspace(*ylim, 200)
    X, Y = np.meshgrid(xs, ys)
    cs = ax.contour(X, Y, f(X, Y), levels=levels, cmap="viridis")
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
    if path is not None:
        path = np.asarray(path)
        ax.plot(path[:, 0], path[:, 1], "o-", color=COLORS[1], ms=3)
        ax.plot(*path[-1], "*", color=COLORS[1], ms=14)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    return ax


def plot_distribution(samples=None, pdf=None, x_range=None, bins=40, ax=None, title=None):
    """Histogram of samples (as a density) with an optional theoretical pdf on top."""
    ax = _ax(ax, (6, 4))
    if samples is not None:
        ax.hist(samples, bins=bins, density=True, alpha=0.5, color=COLORS[0], label="samples")
    if pdf is not None:
        lo, hi = x_range or (np.min(samples), np.max(samples))
        x = np.linspace(lo, hi, 400)
        ax.plot(x, pdf(x), color=COLORS[1], lw=2, label="pdf")
    ax.legend()
    if title:
        ax.set_title(title)
    return ax
