"""Recommended videos, books and papers per module: the single source for README.md and the lesson notebooks.

Each entry: (kind, title, who, url, why, essential)
- kind: "video" | "book" | "paper"
- essential=True -> marked ⭐ ("start here"); the others are optional, for going deeper.
Modules with nothing worth recommending are simply absent.

All links checked 2026-11-24: web pages returned HTTP 200, YouTube IDs were checked through oEmbed, and DOIs through
Crossref (publisher pages often block scripts but open fine in a browser). Add entries here when building new
modules, then run `python tools/update_readme.py`.
"""

def arxiv(id_):
    return f"https://arxiv.org/abs/{id_}"


def yt(id_):
    return f"https://www.youtube.com/watch?v={id_}"


def doi(id_):
    return f"https://doi.org/{id_}"


_3B1B = "https://www.3blue1brown.com/lessons/"
MML = "https://mml-book.github.io/"                       # Deisenroth, Faisal & Ong: free PDF
DLBOOK = "https://www.deeplearningbook.org/contents/"     # Goodfellow, Bengio & Courville: free HTML
VMLS = "https://web.stanford.edu/~boyd/vmls/"             # Boyd & Vandenberghe: free PDF
STRANG_0606 = "https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/"
STRANG_18065 = ("https://ocw.mit.edu/courses/"
                "18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/")

RESOURCES = {
    "00": [
        ("book", "Deep Learning, 'Notation' page", "Goodfellow, Bengio & Courville, 2016", DLBOOK + "notation.html",
         "A one-page cheat sheet of the symbols ML papers use. Keep it open while reading the lesson.", True),
        ("book", "SymPy introductory tutorial", "SymPy docs", "https://docs.sympy.org/latest/tutorials/intro-tutorial/index.html",
         "Go here when you want more symbolic tricks than the lesson shows.", False),
        ("video", "Floating Point Numbers", "Computerphile", yt("PZRI1IfStY0"),
         "Why 0.1 + 0.2 != 0.3, in 9 minutes.", True),
        ("paper", "What Every Computer Scientist Should Know About Floating-Point Arithmetic", "Goldberg, 1991",
         "https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html",
         "The classic reference. Skim the first section on rounding error. The rest is for reference.", False),
    ],
    "01": [
        ("video", "What's so special about Euler's number e?", "3Blue1Brown", _3B1B + "eulers-number",
         "Why e^x is its own derivative, and why ML uses exp and ln everywhere.", True),
        ("book", "Demystifying the Natural Logarithm (ln)", "BetterExplained",
         "https://betterexplained.com/articles/demystifying-the-natural-logarithm-ln/",
         "An intuitive take on ln as 'time needed to grow'.", False),
        ("book", "Deep Learning, ch. 4.1 'Overflow and Underflow'", "Goodfellow, Bengio & Courville, 2016",
         DLBOOK + "numerical.html", "The softmax stability trick in 1 page.", True),
        ("paper", "Accurately computing the log-sum-exp and softmax functions", "Blanchard, Higham & Higham, 2020",
         doi("10.1093/imanum/draa038"), "The careful numerical analysis behind the log-sum-exp trick you implement.", False),
    ],
    "02": [
        ("video", "Vectors (Essence of linear algebra, ch. 1)", "3Blue1Brown", _3B1B + "vectors",
         "The three views of a vector: arrow, list of numbers, and abstract object.", True),
        ("video", "Linear combinations, span, and basis vectors (ch. 2)", "3Blue1Brown", _3B1B + "span",
         "Sets up modules 03–04.", False),
        ("video", "Dot products and duality (ch. 9)", "3Blue1Brown", _3B1B + "dot-products",
         "Why the dot product is a projection.", True),
        ("book", "Mathematics for Machine Learning, ch. 3.1–3.4 (norms, inner products, angles)",
         "Deisenroth, Faisal & Ong, 2020", MML, "The standard ML math textbook. A free PDF.", False),
        ("book", "Introduction to Applied Linear Algebra (VMLS), ch. 1–3", "Boyd & Vandenberghe, 2018", VMLS,
         "Very applied and gentle, built around data vectors.", False),
        ("paper", "Efficient Estimation of Word Representations in Vector Space (word2vec)", "Mikolov et al., 2013",
         arxiv("1301.3781"), "Where 'king − man + woman ≈ queen' and cosine similarity between word vectors became famous.", False),
    ],
    "03": [
        ("video", "Linear transformations and matrices (ch. 3)", "3Blue1Brown", _3B1B + "linear-transformations",
         "The single most important video for this module.", True),
        ("video", "Matrix multiplication as composition (ch. 4)", "3Blue1Brown", _3B1B + "matrix-multiplication",
         "Why matrix multiplication is defined the way it is.", True),
        ("video", "Nonsquare matrices as transformations between dimensions (ch. 8)", "3Blue1Brown",
         _3B1B + "nonsquare-matrices", "How a layer maps 784 dimensions down to 10.", False),
        ("book", "Deep Learning, ch. 2.1–2.3", "Goodfellow, Bengio & Courville, 2016", DLBOOK + "linear_algebra.html",
         "A compact summary of the notation for scalars, vectors, matrices and tensors.", False),
    ],
    "04": [
        ("video", "Inverse matrices, column space and null space (ch. 7)", "3Blue1Brown", _3B1B + "inverse-matrices",
         "Rank and null space as pictures.", True),
        ("video", "The determinant (ch. 6)", "3Blue1Brown", _3B1B + "determinant",
         "The determinant as an area scale factor.", True),
        ("video", "Lecture 2: Elimination with Matrices", "Gilbert Strang, MIT 18.06", yt("QVKj3LADCnA"),
         "Elimination taught by the master.", False),
        ("video", "Lecture 9: Independence, Basis, and Dimension", "Gilbert Strang, MIT 18.06", yt("yjBerM5jWsc"),
         "", False),
        ("video", "Lecture 10: The Four Fundamental Subspaces", "Gilbert Strang, MIT 18.06", yt("nHlE7EgJFds"),
         "The 'big picture' of linear algebra.", False),
    ],
    "05": [
        ("video", "Lecture 15: Projections onto Subspaces", "Gilbert Strang, MIT 18.06", yt("Y_Ac6KiQ1t0"),
         "Derives the normal equation by projection, exactly as the lesson does.", True),
        ("video", "Lecture 17: Orthogonal Matrices and Gram-Schmidt", "Gilbert Strang, MIT 18.06", yt("0MtwqhIwdrI"),
         "", False),
        ("book", "Introduction to Applied Linear Algebra (VMLS), ch. 12–13 (least squares, data fitting)",
         "Boyd & Vandenberghe, 2018", VMLS, "Least squares as the workhorse of data fitting.", True),
        ("book", "Mathematics for Machine Learning, ch. 3.8 (orthogonal projections)", "Deisenroth, Faisal & Ong, 2020",
         MML, "", False),
    ],
    "06": [
        ("video", "Eigenvectors and eigenvalues (ch. 14)", "3Blue1Brown", _3B1B + "eigenvalues",
         "Vectors that only get stretched.", True),
        ("video", "A quick trick for computing eigenvalues (ch. 15)", "3Blue1Brown", _3B1B + "quick-eigen",
         "A shortcut for 2×2 matrices. Useful for the ✍️ exercises.", False),
        ("video", "Change of basis (ch. 13)", "3Blue1Brown", _3B1B + "change-of-basis",
         "Watch it before diagonalization.", False),
        ("video", "Lecture 22: Diagonalization and Powers of A", "Gilbert Strang, MIT 18.06", yt("13r9QY6cmjc"),
         "Why Aᵏ explodes or vanishes, the seed of exploding/vanishing gradients.", False),
        ("video", "Lecture 27: Positive Definite Matrices and Minima", "Gilbert Strang, MIT 18.06", yt("vF7eyJ2g3kU"),
         "Links eigenvalues to the shape of a loss surface (used again in module 10).", False),
        ("paper", "The $25,000,000,000 Eigenvector: The Linear Algebra behind Google", "Bryan & Leise, 2006",
         doi("10.1137/050623280"), "A short, readable paper: PageRank is an eigenvector problem.", True),
        ("paper", "The PageRank Citation Ranking: Bringing Order to the Web", "Page, Brin, Motwani & Winograd, 1999",
         "http://ilpubs.stanford.edu:8090/422/", "The original technical report.", False),
    ],
    "07": [
        ("video", "Singular Value Decomposition (SVD): Overview", "Steve Brunton", yt("gXbThCXjZFM"),
         "Start of an excellent SVD series. Watch the following few videos too.", True),
        ("video", "Principal Component Analysis (PCA), Step-by-Step", "StatQuest", yt("FgakZw6K1QQ"),
         "A gentle PCA walkthrough.", True),
        ("video", "Singular Value Decomposition (the SVD)", "Gilbert Strang, MIT 18.06", yt("mBcLRGuAFUk"), "", False),
        ("book", "Data-Driven Science and Engineering, ch. 1 (SVD)", "Brunton & Kutz, 2019", "http://databookuw.com/",
         "The book that goes with the Brunton videos.", False),
        ("paper", "A Tutorial on Principal Component Analysis", "Shlens, 2014", arxiv("1404.1100"),
         "The clearest PCA paper: derives PCA from both variance and SVD.", True),
        ("paper", "On Lines and Planes of Closest Fit to Systems of Points in Space", "Pearson, 1901",
         doi("10.1080/14786440109462720"), "The paper that introduced PCA. Read it for history.", False),
        ("paper", "The Approximation of One Matrix by Another of Lower Rank", "Eckart & Young, 1936",
         doi("10.1007/BF02288367"), "The theorem behind truncated SVD compression.", False),
        ("paper", "Finding Structure with Randomness: Probabilistic Algorithms for Approximate Matrix Decompositions",
         "Halko, Martinsson & Tropp, 2011", arxiv("0909.4061"), "How sklearn's fast randomized PCA works.", False),
    ],
    "08": [
        ("video", "Attention in transformers, step-by-step (DL ch. 6)", "3Blue1Brown", _3B1B + "attention",
         "Shows the attention matrix ops you build in the lesson.", True),
        ("video", "Transformers, the tech behind LLMs (DL ch. 5)", "3Blue1Brown", _3B1B + "gpt",
         "Embeddings and dot products at scale.", False),
        ("book", "Einsum is All you Need (blog post)", "Tim Rocktäschel, 2018", "https://rockt.ai/2018/04/30/einsum",
         "The best einsum tutorial.", True),
        ("book", "Linear Algebra and Learning from Data / MIT 18.065", "Gilbert Strang, 2019", STRANG_18065,
         "A full course on the matrix methods of ML, with lecture videos.", False),
        ("paper", "Attention Is All You Need", "Vaswani et al., 2017", arxiv("1706.03762"),
         "Read section 3.2: scaled dot-product attention is just the matrix ops from this lesson.", True),
    ],
}

_ICONS = {"video": ("🎥", "Videos"), "book": ("📘", "Books and articles"), "paper": ("📄", "Papers")}


def _ordered(items):
    return sorted(items, key=lambda r: not r[5])   # essential (⭐) first


def _title_md(title, url):
    return f"[{title}]({url})" if url else f"*{title}*"


def reading_cell_markdown(module_key: str) -> str | None:
    """Markdown for the 'Recommended videos, books and papers' cell appended to a lesson notebook."""
    items = RESOURCES.get(module_key, [])
    if not items:
        return None
    lines = ["## 📚 Recommended videos, books and papers",
             "⭐ = start here. The others are optional, for going deeper."]
    for kind, (icon, label) in _ICONS.items():
        group = [r for r in items if r[0] == kind]
        if not group:
            continue
        lines += ["", f"### {icon} {label}"]
        if kind == "paper":
            lines += ["Tip: read the abstract, the figures and the conclusion first, then the method section.", ""]
        for _k, title, who, url, why, ess in _ordered(group):
            lines.append(f"- {'⭐ ' if ess else ''}{_title_md(title, url)} ({who}).{' ' + why if why else ''}")
    return "\n".join(lines)


def readme_lines(module_key: str) -> list[str]:
    """Indented bullet lines inserted under a module in README.md."""
    lines = []
    for kind, (icon, label) in _ICONS.items():
        group = [r for r in RESOURCES.get(module_key, []) if r[0] == kind]
        if group:
            lines.append(f"  - {icon} **{label}**")
            for _k, title, who, url, _why, ess in _ordered(group):
                lines.append(f"    - {'⭐ ' if ess else ''}{_title_md(title, url)} ({who})")
    return lines
