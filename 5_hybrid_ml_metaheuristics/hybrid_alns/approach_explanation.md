# Hybrid ALNS for the One-Dimensional Bin Packing Problem

## I. Formal Problem Statement

### I.1 Optimization problem and input model

The **one-dimensional bin packing problem (1D-BPP)** is defined as follows. We are given:
- a finite set of **items** indexed by $\mathcal{I} = \{1,2,\ldots,n\}$,
- strictly positive integer **sizes** $s_i \in \mathbb{Z}_{>0}$ for each $i \in \mathcal{I}$,
- identical **bins**, each of integer **capacity** $C \in \mathbb{Z}_{>0}$, with the assumption that $s_i \le C$ for all $i$ (otherwise no feasible solution exists).

A **feasible solution** is a partition of $\mathcal{I}$ into a collection of $m$ non-empty subsets (bins) $B_1, B_2, \ldots, B_m$ satisfying:

$$
\biguplus_{j=1}^{m} B_j = \mathcal{I},
\qquad
\sum_{i \in B_j} s_i \le C \quad \forall j \in \{1,\ldots,m\}.
$$

The **objective** is to minimize the number of bins used:

$$
\min_{B_1,\ldots,B_m} \; m.
$$

### I.2 Lower bound

A canonical **continuous relaxation lower bound** is:

$$
\mathrm{LB}_1 = \left\lceil \frac{\sum_{i=1}^{n} s_i}{C} \right\rceil.
$$

**Validity.** Any feasible packing places all item volume into bins of capacity $C$, so total used capacity is at least $\sum_i s_i$. Since each of the $m$ bins contributes at most $C$ units of capacity, we need $mC \ge \sum_i s_i$, i.e., $m \ge \sum_i s_i / C$. Because $m$ is integer, the ceiling follows. $\square$

The bound can be sharpened (e.g. via the $L_2$ bound of Martello and Toth, which accounts for large items that cannot coexist in a bin), but $\mathrm{LB}_1$ is sufficient as a progress indicator in this implementation.

### I.3 Computational complexity

1D-BPP is **strongly NP-hard** (by reduction from 3-Partition). This implies that, unless P = NP, no polynomial-time algorithm can solve all instances optimally. In practice:
- **Exact methods** (branch-and-bound, branch-and-price, column generation) are feasible for small-to-medium instances (up to a few hundred items) but scale poorly.
- **Approximation algorithms** (e.g. First-Fit Decreasing, Best-Fit Decreasing) run in $O(n \log n)$ and achieve a bounded approximation ratio: FFD is known to use at most $\frac{11}{9}\,\mathrm{OPT} + \frac{6}{9}$ bins (Johnson, 1973); BFD achieves the same asymptotic ratio of $\frac{11}{9}$ but the tight additive constant has not been established for BFD specifically.
- **Metaheuristics** sacrifice optimality guarantees in exchange for scalable search over large instances, which is the focus of this solver.

## II. Constructive Initialization: Best-Fit Decreasing

Before ALNS begins, the solver builds a deterministic warm start using **Best-Fit Decreasing (BFD)**.

1. Sort items by non-increasing size (ties by item index).
2. For each item, evaluate all currently open bins that can fit it.
3. Place the item into the feasible bin with **minimum post-placement slack**.
4. If no feasible bin exists, open a new bin.

This is the exact strategy currently used by `_build_ffd_start_solution` (name kept for backward compatibility, implementation is BFD).

Why this matters: a tighter warm start usually gives ALNS fewer bins to improve from, which tends to improve final quality and reduce time spent recovering from weak initial packings.

## III. Metaheuristic Framework: Large-Neighborhood Search and its Adaptive Variant

### III.1 Why local search alone is insufficient

Classical local search improves a solution by applying small *moves* (e.g., relocating a single item). The **neighborhood** of a solution $x$ is the set of solutions reachable by one move. While efficient, small-neighborhood local search suffers from **local optima**: solutions from which no single move improves the objective, yet which are far from globally optimal. For 1D-BPP, local optima are dense and structured around specific item groupings.

### III.2 Large-Neighborhood Search (LNS)

LNS, introduced by Shaw (1998), addresses the local optima problem by operating on *large* implicit neighborhoods. Each iteration consists of two phases:

1. **Destroy**: partially deconstruct the current solution by removing a subset of items from their assigned bins. This produces a *partial solution* $\hat{x}$ and a set of **displaced items** $D$.
2. **Repair**: reinsert all displaced items into $\hat{x}$ to restore feasibility, producing a new complete solution $x'$.

The size of the neighborhood is implicitly exponential in $|D|$, because repair can produce any feasible completion of $\hat{x}$. This allows LNS to escape local optima that are unreachable by any sequence of small moves.

### III.3 Adaptive Large-Neighborhood Search (ALNS)

Ropke and Pisinger (2006) extended LNS by maintaining a **portfolio of destroy operators** and **adapting their selection probabilities** based on observed performance. The rationale is that no single destroy strategy dominates on all instances or at all stages of the search: random destruction explores broadly early on, while targeted destruction (e.g., removing poorly packed bins) is more effective near a local optimum.

In this solver, three destroy operators are available:

| Arm | Operator         | Mechanism                                                                                                                                          |
| --- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0   | **Random**       | Select bins in random order, displacing their items until at least $k$ items are displaced in total.                                               |
| 1   | **Worst-load**   | Greedily select least-loaded bins in order, displacing their items until at least $k$ items are displaced in total.                                |
| 2   | **Related-item** | Select a seed item at random, then displace the seed plus the $k-1$ items with sizes closest to the seed's size. Actual displacement is $\min(k,\, | \text{items} | -1)$ to ensure at least one item always remains placed. |

The target number of displaced items $k$ is sampled uniformly from $[k_{\min}, k_{\max}]$ with:

$$
k_{\min} = \max\!\left(1, \left\lfloor 0.05\,n \right\rfloor\right), \qquad k_{\max} = \max\!\left(k_{\min}+1, \left\lfloor 0.25\,n \right\rfloor\right),
$$

so between 5% and 25% of items are displaced per iteration.

Operator selection is governed by Thompson Sampling (Section V.1).

## IV. Acceptance Mechanism: Simulated Annealing with Plateau Control

### IV.1 Base SA acceptance

Let $x$ be the incumbent and $x'$ the candidate after destroy-repair. With

$$\Delta = f(x') - f(x),$$

the solver accepts when:

$$
\Delta \le 0 \quad \text{or} \quad U < \exp(-\Delta/T),
$$

where $U \sim \text{Uniform}(0,1)$ and $T$ is temperature.

### IV.2 Cooling + reheating used in the current implementation

The solver applies geometric cooling each iteration: $T \leftarrow \alpha T$ with $0<\alpha\le 1$.
In addition, if search stagnates (no best-solution improvement for a plateau window), a **soft reheat** is applied:

$$
T \leftarrow \max(T, 0.35\,T_0).
$$

This restores moderate exploration without fully resetting the annealing schedule.

### IV.3 Early stop on long stagnation

To avoid spending runtime on a flat tail, the run ends early when consecutive non-improving iterations reach a preset limit (`no_improve_limit`).
This improves time efficiency while preserving solution quality in typical runs.

## V. Two Machine Learning Components

The solver uses two machine-learning components:
1. **Adaptive destroy-operator selection** during ALNS iterations.
2. **Learned repair-bin ranking** for reinserting displaced items.

These components target different decisions and are independent in implementation.

### V.1 Adaptive Operator Selection via Thompson Sampling

At each ALNS iteration, the solver chooses one destroy operator from:
- Random
- Worst-load
- Related-item

This is modeled as a 3-arm Bernoulli bandit with Thompson Sampling.

**Per-iteration procedure:**
1. For each operator $k$, sample $\tilde{\theta}_k \sim \mathrm{Beta}(\alpha_k, \beta_k)$.
2. Choose $k^* = \arg\max_k \tilde{\theta}_k$.
3. Apply operator $k^*$, run repair, and evaluate acceptance.
4. Map outcome to reward levels: new-best $\to 1.0$, accepted non-improving $\to 0.5$, rejected $\to 0.0$.
5. Convert reward to Bernoulli feedback by sampling $\tilde r \sim \mathrm{Bernoulli}(r)$, then update only the selected arm:
   - success: $\alpha_{k^*} \leftarrow \alpha_{k^*} + 1$
   - failure: $\beta_{k^*} \leftarrow \beta_{k^*} + 1$

All arms start from the uniform prior $\mathrm{Beta}(1,1)$, so no operator is preferred initially.

### V.2 Machine-Learned Repair: Behavioral Cloning from BFD

After destruction, each displaced item $i$ must be reinserted. Let
$\mathcal{F}(i) = \{j : \text{bin } j \text{ has sufficient residual capacity for item } i\}$
be the feasible-bin set. Repair selects one bin from $\mathcal{F}(i)$, or opens a new bin if $\mathcal{F}(i)=\emptyset$.

The model is trained as binary classification on feasible $(\text{item},\text{bin})$ pairs: predict whether a bin is the preferred placement. Items are processed in non-increasing size order.

#### V.2.1 Feature representation

Each feasible $(i, j)$ pair is encoded as an **11-dimensional feature vector**. All features are normalized by capacity $C$.

| Index | Feature                    | Description                                  |
| ----- | -------------------------- | -------------------------------------------- |
| 0     | $s_i / C$                  | Normalized item size                         |
| 1     | $(s_i / C)^2$              | Squared normalized size                      |
| 2     | $\mathrm{rank}(i) / n$     | Normalized item-size rank                    |
| 3     | $\text{remaining} / n$     | Fraction of not-yet-inserted displaced items |
| 4     | $\ell_j / C$               | Normalized current bin load                  |
| 5     | $(C - \ell_j) / C$         | Residual capacity                            |
| 6     | $(C - \ell_j - s_i) / C$   | Post-placement slack                         |
| 7     | $\|B_j\| / n$              | Normalized bin occupancy (item count)        |
| 8     | $\max_{k \in B_j} s_k / C$ | Largest item already in bin                  |
| 9     | $\min_{k \in B_j} s_k / C$ | Smallest item already in bin                 |
| 10    | $s_i / (C - \ell_j)$       | Fill ratio                                   |

**Feature contract:** training features (`train_repair_model.py::_make_features`) and inference features (`solver.py::_make_repair_features`) must match exactly. The serialized model bundle includes `feature_version` so the solver can reject incompatible bundles.

#### V.2.2 Model architecture

The repair ranker is a `GradientBoostingClassifier` (gradient-boosted decision trees).
A `StandardScaler` is fit and stored with the bundle for compatibility.
At inference time, the solver scores all feasible bins with `predict_proba` and chooses the bin with maximum predicted probability.

Balanced sample weights are used to reduce bias from class imbalance caused by negative subsampling. The main held-out metric is ROC-AUC because the solver consumes probability scores for ranking.

#### V.2.3 Offline training data generation

Training rows are generated from two trace types:

- **Fresh BFD traces:** replay BFD from scratch. For each insertion step with multiple feasible bins, label BFD's chosen bin as positive and sampled alternatives as negative.
- **Post-destruction repair traces:** build a full BFD solution, evict a random fraction of bins, then reinsert displaced items into the surviving partial solution using the same labeling rule.

Pooling both trace types reduces distribution shift between offline training and online ALNS repair states.
