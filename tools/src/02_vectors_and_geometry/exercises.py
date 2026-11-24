# %% [markdown]
# # Module 02: Exercises (Vectors and Geometry)
# 💻 = code, ✍️ = by hand first, then type the answer. Solutions are in `solutions.ipynb`.

# %%
#! SETUP

# %% [markdown]
# ### Exercise 1 ✍️: Vector arithmetic
# $\mathbf{u} = (2, -1, 3)$, $\mathbf{v} = (1, 4, 0)$. Compute:
# 1. $\mathbf{u} + \mathbf{v}$
# 2. $3\mathbf{u} - \mathbf{v}$
# 3. $\mathbf{u} \cdot \mathbf{v}$
# 4. $\lVert\mathbf{u}\rVert_2^2$

# %%
### BEGIN SOLUTION
#> a1 = a2 = a3 = a4 = None  # TODO
a1 = [3, 3, 3]
a2 = [5, -7, 9]
a3 = -2           # 2*1 + (-1)*4 + 3*0
a4 = 14           # 4 + 1 + 9
### END SOLUTION

# %%
u, v = np.array([2, -1, 3]), np.array([1, 4, 0])
check(a1, u + v, name="u + v")
check(a2, 3 * u - v, name="3u − v")
check(a3, u @ v, name="u · v")
check(a4, u @ u, name="‖u‖²")

# %% [markdown]
# ### Exercise 2 ✍️: Angles and orthogonality
# 1. Are $(1, 2)$ and $(-4, 2)$ orthogonal? (`True` / `False`)
# 2. The angle in degrees between $(1, 0)$ and $(1, 1)$?
# 3. The angle in degrees between $(1, 1)$ and $(-2, -2)$?
# 4. Is the dot product of $(3, 1)$ and $(-1, 2)$ positive, zero or negative? Answer `"positive"`, `"zero"`
#    or `"negative"`, then say the angle is acute, right or obtuse to yourself.

# %%
### BEGIN SOLUTION
#> b1 = b2 = b3 = b4 = None  # TODO
b1 = True         # 1*(-4) + 2*2 = 0
b2 = 45
b3 = 180          # opposite directions
b4 = "negative"   # -3 + 2 = -1: an obtuse angle
### END SOLUTION

# %%
ang = lambda p, q: np.degrees(np.arccos(p @ q / (np.linalg.norm(p) * np.linalg.norm(q))))
check(b1, bool(np.array([1, 2]) @ np.array([-4, 2]) == 0), name="orthogonal?")
check(b2, ang(np.array([1, 0]), np.array([1, 1])), tol=1e-4, name="angle 1")
check(b3, ang(np.array([1, 1]), np.array([-2, -2])), tol=1e-4, name="angle 2")
check(b4, "negative", name="sign")

# %% [markdown]
# ### Exercise 3 ✍️: Projections
# Use $\text{proj}_{\mathbf{v}}\mathbf{w} = \frac{\mathbf{v}\cdot\mathbf{w}}{\mathbf{v}\cdot\mathbf{v}}\mathbf{v}$.
# 1. Project $\mathbf{w} = (2, 3)$ onto $\mathbf{v} = (1, 0)$.
# 2. Project $\mathbf{w} = (2, 3)$ onto $\mathbf{v} = (1, 1)$.
# 3. In case 2, what is the residual $\mathbf{w} - \text{proj}_{\mathbf{v}}\mathbf{w}$?

# %%
### BEGIN SOLUTION
#> c1 = c2 = c3 = None  # TODO
c1 = [2, 0]
c2 = [2.5, 2.5]       # (5/2) * (1, 1)
c3 = [-0.5, 0.5]      # orthogonal to (1, 1), as it should be
### END SOLUTION

# %%
pj = lambda w_, v_: (v_ @ w_) / (v_ @ v_) * v_
check(c1, pj(np.array([2.0, 3]), np.array([1.0, 0])), name="proj 1")
check(c2, pj(np.array([2.0, 3]), np.array([1.0, 1])), name="proj 2")
check(c3, np.array([2.0, 3]) - pj(np.array([2.0, 3]), np.array([1.0, 1])), name="residual")

# %% [markdown]
# ### Exercise 4 ✍️: Norms
# For $\mathbf{x} = (1, -2, 2)$ compute the L1, L2 and L∞ norms, and the unit vector $\mathbf{x}/\lVert\mathbf{x}\rVert_2$.

# %%
### BEGIN SOLUTION
#> n1 = n2 = n3 = n4 = None  # TODO
n1 = 5
n2 = 3            # sqrt(1 + 4 + 4)
n3 = 2
n4 = [1 / 3, -2 / 3, 2 / 3]
### END SOLUTION

# %%
xx = np.array([1.0, -2, 2])
check(n1, np.linalg.norm(xx, 1), name="L1")
check(n2, np.linalg.norm(xx), name="L2")
check(n3, np.linalg.norm(xx, np.inf), name="L∞")
check(n4, xx / np.linalg.norm(xx), name="unit vector")

# %% [markdown]
# ### Exercise 5 ✍️: Span
# 1. Is $(4, 6)$ in the span of $\{(2, 3)\}$?
# 2. Is $(1, 0)$ in the span of $\{(2, 3)\}$?
# 3. Do $(1, 2)$ and $(2, 4)$ span all of $\mathbb{R}^2$?
# 4. Do $(1, 2)$ and $(2, 1)$ span all of $\mathbb{R}^2$?

# %%
### BEGIN SOLUTION
#> s1 = s2 = s3 = s4 = None  # TODO: True / False
s1 = True         # 2 * (2, 3)
s2 = False        # every multiple of (2, 3) has y = 1.5x
s3 = False        # (2, 4) = 2 * (1, 2): both lie on one line
s4 = True         # different directions
### END SOLUTION

# %%
check(s1, True, name="(4,6) in span"); check(s2, False, name="(1,0) in span")
check(s3, False, name="(1,2),(2,4) span R²"); check(s4, True, name="(1,2),(2,1) span R²")

# %% [markdown]
# ### Exercise 6 💻: Projection and decomposition
# Write `decompose(w, v)` that returns `(parallel, perpendicular)`, where `parallel` is the projection of w onto v and
# `perpendicular = w - parallel`. It must work in any dimension.

# %%
def decompose(w, v):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    parallel = (v @ w) / (v @ v) * v
    return parallel, w - parallel
    ### END SOLUTION


# %%
w6, v6 = rng.normal(size=10), rng.normal(size=10)
par, perp = decompose(w6, v6)
check(par + perp, w6, name="parallel + perpendicular = w")
check(perp @ v6, 0.0, tol=1e-9, name="perpendicular · v")
check(np.abs(par @ v6), np.linalg.norm(par) * np.linalg.norm(v6), name="parallel ∥ v")

# %% [markdown]
# ### Exercise 7 💻: Semantic search with cosine similarity
# `docs` is a (n_docs, d) matrix of embeddings, one row per document.
# 1. `cosine_sim_matrix(A, B)` returns the (len(A), len(B)) matrix of cosine similarities between rows. No loops:
#    normalize the rows, then take one matrix product.
# 2. `top_k(query, docs, k)` returns the indices of the k most similar documents, most similar first.

# %%
def cosine_sim_matrix(A, B):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    An = A / np.linalg.norm(A, axis=1, keepdims=True)
    Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
    return An @ Bn.T
    ### END SOLUTION


def top_k(query, docs, k):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    sims = cosine_sim_matrix(query[None, :], docs)[0]
    return np.argsort(-sims)[:k]
    ### END SOLUTION


# %%
from scipy.spatial.distance import cdist
docs = rng.normal(size=(100, 16))
q = docs[42] * 3 + rng.normal(scale=0.1, size=16)     # a scaled, noisy copy of doc 42
check(cosine_sim_matrix(docs[:5], docs[5:9]), 1 - cdist(docs[:5], docs[5:9], "cosine"), name="cosine matrix")
check(top_k(q, docs, 1)[0], 42, name="best match")
check(top_k(q, docs, 5), np.argsort(cdist(q[None], docs, "cosine")[0])[:5], name="top-5 order")

# %% [markdown]
# ### Exercise 8 💻: Vectorized 1-nearest-neighbor classifier
# 1. `pairwise_sq_dists(A, B)` returns the squared Euclidean distances between every row of A and every row of B,
#    with shape (len(A), len(B)). Use the ‖a‖² − 2a·b + ‖b‖² trick from the lesson. No loops.
# 2. `predict_1nn(X_train, y_train, X_test)` labels each test point with the label of its closest training point.

# %%
def pairwise_sq_dists(A, B):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    a2 = (A ** 2).sum(axis=1)[:, None]
    b2 = (B ** 2).sum(axis=1)[None, :]
    return np.maximum(a2 - 2 * A @ B.T + b2, 0)
    ### END SOLUTION


def predict_1nn(X_train, y_train, X_test):
    ### BEGIN SOLUTION
    #> raise NotImplementedError
    return y_train[np.argmin(pairwise_sq_dists(X_test, X_train), axis=1)]
    ### END SOLUTION


# %%
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
Xb, yb = make_blobs(300, centers=3, random_state=0)
Xtr, ytr, Xte = Xb[:200], yb[:200], Xb[200:]
check(pairwise_sq_dists(Xte[:4], Xtr[:6]), cdist(Xte[:4], Xtr[:6], "sqeuclidean"), name="squared distances")
check(predict_1nn(Xtr, ytr, Xte), KNeighborsClassifier(1).fit(Xtr, ytr).predict(Xte), name="1-NN predictions")
