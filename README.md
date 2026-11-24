# Math Foundations for AI/ML, Step by Step

A hands-on course covering the math behind machine learning and deep learning: notation, linear algebra,
calculus, probability, statistics, information theory and optimization. Each idea is built up through intuition and
pictures, then checked in code.

It goes with two sibling courses:
- **Classical ML**: [`classical-machine-learning-basic-to-advanced-in-steps`](https://github.com/suraj9321/classical-machine-learning-basic-to-advanced-in-steps)
- **Deep learning**: [`deep-learning-cv-nlp-llm-zero-to-hero`](https://github.com/suraj9321/deep-learning-cv-nlp-llm-zero-to-hero)

The [map below](#which-course-needs-what) shows which math module each of their modules relies on.

## Phases
- [Phase 0: Tools and notation](#phase-0-tools-and-notation)
- [Phase 1: Linear algebra](#phase-1-linear-algebra)
- [Phase 2: Calculus](#phase-2-calculus)
- [Phase 3: Probability](#phase-3-probability)
- [Phase 4: Statistics](#phase-4-statistics)
- [Phase 5: Information theory](#phase-5-information-theory)
- [Phase 6: Optimization](#phase-6-optimization)
- [Phase 7: Capstone](#phase-7-capstone)

## How each module works
Each folder in `modules/` contains:

| File | What to do |
|---|---|
| `lesson.ipynb` | Read it and run every cell. Change the numbers and re-run to build intuition. |
| `exercises.ipynb` | Replace each `None` / `raise NotImplementedError` with your work, then run the **check** cell below it (✅ = pass). |
| `solutions.ipynb` | Open it only after a real attempt, or to compare approaches. |

The teaching pattern is **intuition → picture → a little math → verify in code → where it shows up in ML/DL**.

There are two kinds of exercise:
- 💻 **Code**: implement a formula or algorithm in NumPy. The asserts check it.
- ✍️ **By hand**: work it out on paper first, *then* type your answer, e.g. `answer = [3, 7]`, and run
  `check(answer, ...)`. Don't compute these in Python: building the skill by hand is the point.

Each lesson ends with **"Where you'll use this"** (links to the ML/DL courses) and **📚 recommended videos, books and
papers**. ⭐ marks the ones to start with.

## Setup (one time)
```bash
cd ai-ml-math-foundation-zero-to-hero
python3.12 -m venv .venv            # already created for you
source .venv/bin/activate
pip install -r requirements.txt     # already installed
jupyter lab                         # opens in your browser
```
In VS Code, open a notebook and pick the `.venv` Python as the kernel.

The first code cell of each notebook makes these available: `np`, `plt`, `sp` (SymPy), `check` (from `common/checks.py`),
`mplt` (the plotting helpers in `common/plotting.py`) and `rng` (a seeded random generator).

### For maintainers: how notebooks are built
Notebooks are generated from plain-Python sources in `tools/src/<module>/` (`lesson.py`, `exercises.py`):
```bash
.venv/bin/python tools/build.py 00 01 --execute   # build + run lesson and solutions to verify them
.venv/bin/python tools/update_readme.py           # refresh the 📚 lists below from tools/resources.py
```

## Roadmap & progress
Tick each box as you finish a module. ✅ built · ⏳ coming next

### Phase 0: Tools and notation
- [ ] **00 Math with Python and notation** ✅
  - Reading math notation: Σ, Π, ∈, ℝⁿ, subscripts vs superscripts x⁽ⁱ⁾, ŷ, ‖x‖, argmin/argmax, ∀, ∃, :=, ∝
  - Turning formulas into NumPy: sums, products, nested sums, indicator functions
  - Functions f: ℝⁿ → ℝ and plotting them
  - SymPy: symbols, simplify, expand, solve, substitute, lambdify
  - Floating point: why `0.1 + 0.2 != 0.3`, `np.isclose`, machine epsilon, overflow, underflow, catastrophic cancellation
- [ ] **01 Functions, exponents and logs** ✅
  - Functions, domain and range, composition, inverse functions
  - Polynomials, exponents, and exp growth
  - Logarithm rules, and why ML loves logs (products → sums, log-likelihood)
  - Sigmoid and its inverse (logit); softmax and temperature
  - Numerical stability: log-sum-exp, a stable softmax, `log1p`/`expm1`

### Phase 1: Linear algebra
- [ ] **02 Vectors and geometry** ✅
  - Vectors as arrows and as data points; addition and scaling
  - Linear combinations and span
  - Dot product: algebra, geometry (angles) and projection
  - Norms: L1, L2, L∞; unit vectors; distances
  - Cosine similarity and embeddings (a toy word-vector example)
  - Orthogonality
- [ ] **03 Matrices as transformations** ✅
  - A matrix as a function: what happens to e₁ and e₂
  - Rotation, scaling, shear, reflection and projection matrices, visualized
  - Matrix–vector product: row view (dot products) vs column view (combination of columns)
  - Matrix multiplication as composition; why AB ≠ BA
  - Transpose, symmetric matrices, the shape rules, `X @ W` on a batch
  - A neural-network layer as a matrix transformation
- [ ] **04 Linear systems, rank and inverses** ✅
  - Ax = b: three geometric pictures (rows, columns, transformation)
  - Gaussian elimination by hand and in code
  - Linear independence, basis, dimension, rank
  - Column space and null space
  - The determinant as an area/volume scale factor
  - Inverse matrices, `np.linalg.solve` vs `inv`, singular and near-singular matrices
- [ ] **05 Orthogonality and least squares** ✅
  - Projecting onto a line and onto a subspace
  - Orthonormal bases and orthogonal matrices (they preserve lengths)
  - Gram–Schmidt and QR decomposition
  - Least squares: derive the normal equations XᵀXw = Xᵀy from geometry
  - Fitting lines and polynomials; solving with QR vs the normal equations
- [ ] **06 Eigenvalues and eigenvectors** ⏳
  - Eigenvectors as directions that only stretch
  - The characteristic polynomial (2×2 by hand); trace and determinant
  - Diagonalization and matrix powers Aᵏ (exploding/vanishing behavior)
  - Symmetric matrices: real eigenvalues, orthogonal eigenvectors (the spectral theorem)
  - Quadratic forms and positive (semi)definite matrices; the shape of a loss bowl
  - Power iteration and PageRank
- [ ] **07 SVD and PCA** ⏳
  - SVD: rotate → stretch → rotate
  - Relation to eigen-decomposition of AᵀA
  - Low-rank approximation (Eckart–Young) and image compression
  - Pseudo-inverse and least squares via SVD; condition number
  - PCA from scratch: centering, covariance, principal directions, explained variance
  - PCA via SVD, checked against scikit-learn
- [ ] **08 Linear algebra for deep learning** ⏳
  - Tensors: shapes like (batch, seq, features) and (N, C, H, W)
  - Broadcasting rules in depth
  - `np.einsum`: dot, outer, matmul, batched matmul, trace, transpose
  - Matrix norms (Frobenius, spectral) and why they matter for stability
  - Scaled dot-product attention written as pure matrix ops

### Phase 2: Calculus
- [ ] **09 Limits, derivatives and Taylor series** ⏳
  - Limits and continuity; the derivative as a limit and as a slope
  - Derivative rules (power, product, quotient, chain), checked with SymPy
  - Derivatives of exp, log, sigmoid, tanh, ReLU
  - Numerical differentiation: forward vs central differences, choosing h
  - Taylor approximations: linear and quadratic, and why gradient descent works
- [ ] **10 Multivariable calculus** ⏳
  - Partial derivatives and the gradient; the gradient points uphill
  - Directional derivatives; contour plots and level sets
  - The Jacobian (vector → vector functions)
  - The Hessian, curvature, and classifying critical points (min, max, saddle)
  - Multivariable Taylor expansion
- [ ] **11 Chain rule and backpropagation** ⏳
  - Computational graphs
  - The multivariable chain rule
  - Forward mode vs reverse mode differentiation
  - Build a tiny autograd engine (scalar `Value` class) and train a neuron with it
- [ ] **12 Matrix calculus** ⏳
  - Derivatives with respect to vectors and matrices; shape-checking as a debugging tool
  - Layout conventions (numerator vs denominator)
  - Standard results: ∇(aᵀx), ∇(xᵀAx), ∇‖Xw − y‖²
  - Gradients of softmax + cross-entropy and of a linear layer
  - Gradient checking against finite differences
- [ ] **13 Integrals and expectations** ⏳
  - The integral as area and as accumulation; the fundamental theorem of calculus
  - Numerical integration (Riemann sums, trapezoid, `scipy.integrate.quad`)
  - Integrals in probability: normalizing a density, expectation as an integral
  - Monte Carlo integration and its error rate

### Phase 3: Probability
- [ ] **14 Probability basics** ⏳
  - Sample spaces, events, the axioms
  - Counting: permutations and combinations
  - Conditional probability, independence, the law of total probability
  - Bayes' rule with base rates (medical tests, spam filters)
- [ ] **15 Random variables and distributions** ⏳
  - Discrete vs continuous; PMF, PDF, CDF
  - Expectation, variance, and linearity of expectation
  - Bernoulli, binomial, categorical, Poisson, uniform, normal, exponential, beta
  - Sampling from distributions (inverse-CDF sampling)
- [ ] **16 Multivariate distributions** ⏳
  - Joint, marginal and conditional distributions
  - Covariance and correlation; the covariance matrix
  - The multivariate Gaussian and its geometry (ellipses = eigenvectors of Σ)
  - Sampling correlated data via the Cholesky decomposition
- [ ] **17 Limit theorems and sampling** ⏳
  - Law of large numbers
  - Central limit theorem: simulated and visualized
  - Standard error; sampling distributions
  - Bootstrap resampling

### Phase 4: Statistics
- [ ] **18 Statistical inference** ⏳
  - Estimators, bias, variance, mean squared error
  - Maximum likelihood estimation (MLE), by hand and numerically
  - Confidence intervals
  - Hypothesis tests, p-values, and common misreadings; an A/B test
- [ ] **19 Bayesian inference** ⏳
  - Prior, likelihood, posterior
  - Conjugate pairs (beta-binomial, normal-normal)
  - MAP estimation, and why L2 regularization = a Gaussian prior
  - The posterior predictive distribution

### Phase 5: Information theory
- [ ] **20 Information theory** ⏳
  - Information content and entropy
  - Cross-entropy and why it is the classification loss
  - KL divergence (asymmetry, forward vs reverse)
  - Mutual information
  - Perplexity of language models

### Phase 6: Optimization
- [ ] **21 Convexity and optimization** ⏳
  - Convex sets and convex functions; why convex = no bad local minima
  - Gradient descent: step size, convergence, and the condition number
  - Newton's method
  - Constrained optimization and Lagrange multipliers (link to SVMs and max-entropy)
- [ ] **22 Stochastic optimization** ⏳
  - Stochastic and mini-batch gradient descent; gradient noise
  - Momentum, RMSProp and Adam from scratch, compared on hard loss surfaces
  - Learning-rate schedules and warmup
  - Saddle points and plateaus in high dimensions

### Phase 7: Capstone
- [ ] **23 Capstone: from math to models** ⏳
  - Linear regression three ways: normal equations, QR/SVD, gradient descent
  - Logistic regression derived from MLE, trained with your own optimizer
  - PCA via SVD on a real dataset
  - A 2-layer neural network with hand-derived backprop, checked by gradient checking

## Which course needs what
| Math module(s) | Classical ML course | Deep learning course |
|---|---|---|
| 00–01 notation, functions, logs | all modules; 08 logistic regression | all modules; 01 MLP |
| 02–05 vectors, matrices, least squares | 07 linear regression, 09 regularization, 11 KNN | 00 tensors, 01 MLP |
| 06–07 eigen, SVD, PCA | 16 dimensionality reduction | 15 autoencoders |
| 08 linear algebra for DL | n/a | 07–08 convolutions, 14 ViT, 23–25 attention & GPT |
| 09–12 calculus, backprop | 02 gradient descent, 07–08 regression | 02 backprop, 03 autograd, 05 optimization |
| 13–17 integrals, probability | 03 probability, 11 naive Bayes | 16 VAEs, 19–20 diffusion |
| 18–19 statistics, Bayes | 06 ML fundamentals, 09 cross-validation, 17 tuning | 06 regularization |
| 20 information theory | 08 log-loss, 12 decision trees (entropy) | 04 cross-entropy, 16 VAEs (KL), 25 perplexity |
| 21–22 optimization | 07 gradient descent, 14 SVM | 05 optimizers, 27 fine-tuning |

## Suggested pace
About 3–5 hours per module: about 1 hour for the lesson, 1–2 hours for exercises, and the rest re-doing what felt shaky.
You don't need to finish this course before starting the others. Many people run it **alongside** them: when an
ML/DL module uses some math, do the matching math module first (see the map above).

## Tips
- Do the ✍️ exercises on actual paper. Speed with small cases (2×2 matrices, 1-D derivatives) is what makes papers readable.
- Before every matrix operation, say the **shapes** out loud. Most bugs are shape bugs.
- When a formula looks scary, plug in tiny numbers and compute one term by hand.
- If you're stuck on an exercise for more than 20 minutes, re-read the matching lesson section before opening the solution.
- Commit your progress with git after each module: `git add -A && git commit -m "finished module 02"`.

## License
Released under the [MIT License](LICENSE). The recommended videos, books and papers linked in the lessons belong to their
respective authors and are not covered by this license.
