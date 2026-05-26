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

A tighter warm start gives ALNS fewer bins to improve from, which tends to improve final solution quality and reduces time spent recovering from a weak initial packing.

## III. Metaheuristic Framework: Large-Neighborhood Search and its Adaptive Variant

### III.1 Why local search alone is insufficient

Classical local search improves a solution by applying small *moves* (e.g., relocating a single item). The **neighborhood** of a solution $x$ is the set of solutions reachable by one move. While efficient, small-neighborhood local search suffers from **local optima**: solutions from which no single move improves the objective, yet which are far from globally optimal. For 1D-BPP, local optima are dense and structured around specific item groupings.

### III.2 Large-Neighborhood Search (LNS)

LNS, introduced by Shaw (1998), addresses the local optima problem by operating on *large* implicit neighborhoods. Each iteration consists of two phases:

1. **Destroy**: partially deconstruct the current solution by removing a subset of items from their assigned bins. This produces a *partial solution* $\hat{x}$ and a set of **displaced items** $D$.
2. **Repair**: reinsert all displaced items into $\hat{x}$ to restore feasibility, producing a new complete solution $x'$.

The size of the neighborhood is implicitly exponential in $|D|$, because repair can produce any feasible completion of $\hat{x}$. This allows LNS to escape local optima that are unreachable by any sequence of small moves.

### III.3 Adaptive Large-Neighborhood Search (ALNS)

Ropke and Pisinger (2006) extended LNS by maintaining a **portfolio of destroy operators** and adapting their selection based on observed performance. The rationale is that no single destroy strategy dominates on all instances or at all stages of the search: random destruction explores broadly early on, while targeted destruction is more effective near a local optimum.

In this solver, three destroy operators are available:

| Arm | Operator         | Mechanism                                                                                                                                                                                                                          |
| --- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0   | **Random**       | Sample exactly $\min(k,\, n-1)$ items uniformly at random across all placed items and remove them from their bins.                                                                                                                 |
| 1   | **Worst-load**   | Sort bins by ascending load (with a small uniform tie-breaking perturbation). Collect items from bins in that order until $\min(k,\, n-1)$ items have been selected, then remove them.                                             |
| 2   | **Related-item** | Select a seed item uniformly at random. Sort all other placed items by ascending absolute size difference from the seed. Remove the seed plus the $\min(k-1,\, n-2)$ closest items, for a total of $\min(k,\, n-1)$ removed items. |

All three operators guarantee that at least one item remains placed, preventing total solution destruction.

The target number of displaced items $k$ is drawn uniformly from $[k_{\min},\, k_{\max}]$ at each iteration, with:

$$
k_{\min} = \max\!\left(1, \left\lfloor 0.05\,n \right\rfloor\right), \qquad k_{\max} = \max\!\left(k_{\min}+1, \left\lfloor 0.25\,n \right\rfloor\right),
$$

so between 5% and 25% of items are displaced per iteration. In addition, $k_{\max}$ is expanded adaptively when search stagnates: as `iterations_since_improvement` grows toward `no_improve_limit`, the upper bound interpolates linearly from $k_{\min}$ to $k_{\max}$, broadening the destruction radius to encourage diversification.

Operator selection is governed by a warm-start contextual bandit (Section V.1).

## IV. Acceptance Mechanism: Simulated Annealing with Reheating and Restart

### IV.1 Base SA acceptance

Let $x$ be the current solution and $x'$ the candidate produced after destroy-repair. Define:

$$\Delta = f(x') - f(x).$$

The candidate is accepted when:

$$
\Delta \le 0 \quad \text{or} \quad U < \exp(-\Delta/T),
$$

where $U \sim \mathrm{Uniform}(0,1)$ and $T > 0$ is the current temperature.

### IV.2 Cooling and reheating

The solver applies geometric cooling after every iteration:

$$T \leftarrow \alpha_{\mathrm{cool}} \cdot T, \qquad 0 < \alpha_{\mathrm{cool}} \le 1.$$

To prevent the temperature from collapsing to near zero during long stagnation periods, a **soft reheat** is applied whenever `iterations_since_improvement` is a positive multiple of $\max(50,\, \lfloor\mathrm{no\_improve\_limit}/4\rfloor)$:

$$T \leftarrow \max(T,\; 0.35 \cdot T_0).$$

This restores moderate acceptance of worse solutions without fully resetting the annealing schedule.

### IV.3 Diversification restart

When `iterations_since_improvement` reaches `no_improve_limit`, the solver performs a **diversification restart**:
1. The current solution is reset to the best solution found so far.
2. The temperature is reheated: $T \leftarrow \max(T,\; 0.20 \cdot T_0)$.
3. The patience window is shrunk: $\mathrm{no\_improve\_limit} \leftarrow \max\!\left(100,\, \lfloor 2\,\mathrm{no\_improve\_limit}/3 \rfloor\right)$, so successive restarts trigger progressively sooner.
4. `iterations_since_improvement` is reset to zero.

The outer iteration budget (`max_iterations`) is the sole hard termination criterion; the restart mechanism never terminates the run.

## V. Two Machine Learning Components

The solver incorporates two machine-learning components that target distinct decisions:

1. **Destroy-operator selection**: a contextual bandit chooses which of the three destroy operators to apply at each iteration.
2. **Repair-bin ranking**: a trained classifier scores feasible bins for each displaced item during repair.

The two components are independent in implementation and can be enabled or disabled separately.

### V.1 Adaptive Operator Selection via Warm-Start LinUCB

At each ALNS iteration the solver selects one of the three destroy operators using a **`WarmStartLinUCBBandit`**, which runs two sequential phases.

#### V.1.1 Warm-up phase: Beta-Bernoulli Thompson Sampling (first 300 calls)

For the first 300 calls, operator selection uses **Beta-Bernoulli Thompson Sampling** (context-free). For each call:

1. Sample $\tilde{\theta}_k \sim \mathrm{Beta}(\alpha_k, \beta_k)$ for each arm $k \in \{0,1,2\}$.
2. Select $k^* = \arg\max_k \tilde{\theta}_k$.
3. After observing the reward $r \in [0,1]$, draw $\tilde{r} \sim \mathrm{Bernoulli}(r)$ and update only arm $k^*$:
   - $\tilde{r} = 1$: $\alpha_{k^*} \leftarrow \alpha_{k^*} + 1$
   - $\tilde{r} = 0$: $\beta_{k^*} \leftarrow \beta_{k^*} + 1$

All arms are initialised at $\mathrm{Beta}(1,1)$ (uniform prior). This phase quickly identifies well-performing operators without requiring the context vector, mitigating the cold-start problem of pure LinUCB on short runs.

#### V.1.2 Exploitation phase: Disjoint LinUCB (calls 301 onward)

After the warm-up, operator selection switches to a **disjoint LinUCB bandit** (Chu et al., ICML 2011) with $\alpha = 0.3$. Each arm $k$ maintains:
- $A_k^{-1} \in \mathbb{R}^{5 \times 5}$: inverse of the gram matrix (initialised to $I$),
- $\mathbf{b}_k \in \mathbb{R}^5$: accumulated reward vector (initialised to $\mathbf{0}$).

**Selection.** Given context vector $\mathbf{x} \in \mathbb{R}^5$, select:

$$k^* = \arg\max_k \left[ \hat{\boldsymbol{\theta}}_k^\top \mathbf{x} + \alpha \sqrt{\mathbf{x}^\top A_k^{-1} \mathbf{x}} \right], \qquad \hat{\boldsymbol{\theta}}_k = A_k^{-1} \mathbf{b}_k.$$

**Update.** After observing reward $r$ for arm $k^*$, update using the Sherman–Morrison rank-1 formula ($O(d^2)$):

$$A_{k^*}^{-1} \leftarrow A_{k^*}^{-1} - \frac{(A_{k^*}^{-1}\mathbf{x})(A_{k^*}^{-1}\mathbf{x})^\top}{1 + \mathbf{x}^\top A_{k^*}^{-1} \mathbf{x}}, \qquad \mathbf{b}_{k^*} \leftarrow \mathbf{b}_{k^*} + r\,\mathbf{x}.$$

#### V.1.3 Context vector

The 5-dimensional context vector $\mathbf{x}$ encodes the current search state:

| Index | Feature                                | Description                                         |
| ----- | -------------------------------------- | --------------------------------------------------- |
| 0     | $T / T_0$                              | Normalised temperature; 1 = start (hot), 0 = cold   |
| 1     | `iter_no_improve` / `no_improve_limit` | Stagnation progress toward next restart             |
| 2     | $f(x) / \mathrm{LB}_1$                 | Ratio of current cost to lower bound; 1.0 = optimal |
| 3     | $k / n$                                | Current destruction radius as a fraction of $n$     |
| 4     | `iteration` / `max_iterations`         | Overall search progress                             |

#### V.1.4 Reward signal

The reward $r \in [0,1]$ passed to whichever phase is active is computed as follows. Let $\mathrm{gap} = \max(1, f(x) - \mathrm{LB}_1)$ be the current gap to the lower bound after the update to the current solution.

$$
r = \begin{cases}
\min\!\left(1,\; \dfrac{\max(0, -\Delta)}{\mathrm{gap}}\right) & \text{if bins were saved } (\Delta < 0) \\
0.2 & \text{if accepted without saving bins} \\
0.0 & \text{if rejected}
\end{cases}
$$

Normalising by the gap rewards bin savings more highly when the solution is already close to the lower bound, reflecting the increasing difficulty of further improvement.

### V.2 Machine-Learned Repair: Behavioral Cloning from BFD

After destruction, each displaced item $i$ must be reinserted. Let:

$$\mathcal{F}(i) = \{j : \ell_j + s_i \le C\}$$

be the set of feasible bins for item $i$, where $\ell_j$ is the current load of bin $j$. If $\mathcal{F}(i) = \emptyset$, a new bin is opened immediately. Otherwise, the repair model scores every bin in $\mathcal{F}(i)$ and places item $i$ in the bin with the highest predicted score. Items are processed in non-increasing size order.

The model is trained as a binary classifier on feasible $(\text{item}, \text{bin})$ pairs: label 1 if BFD would have chosen that bin, label 0 otherwise.

#### V.2.1 Feature representation

Each feasible $(i, j)$ pair is encoded as an **11-dimensional feature vector**. All features are normalised by capacity $C$ or instance size $n$.

| Index | Feature                    | Description                                                     |
| ----- | -------------------------- | --------------------------------------------------------------- |
| 0     | $s_i / C$                  | Normalised item size                                            |
| 1     | $(s_i / C)^2$              | Squared normalised size                                         |
| 2     | $\mathrm{rank}(i) / n$     | Size rank of item $i$ among all $n$ items (0 = largest)         |
| 3     | $\mathrm{remaining} / n$   | Fraction of displaced items not yet reinserted                  |
| 4     | $\ell_j / C$               | Normalised current bin load                                     |
| 5     | $(C - \ell_j) / C$         | Residual capacity fraction                                      |
| 6     | $(C - \ell_j - s_i) / C$   | Post-placement slack fraction                                   |
| 7     | $\lvert B_j \rvert / n$    | Normalised bin occupancy (item count)                           |
| 8     | $\max_{k \in B_j} s_k / C$ | Largest item already in bin $j$                                 |
| 9     | $\min_{k \in B_j} s_k / C$ | Smallest item already in bin $j$                                |
| 10    | $s_i / (C - \ell_j)$       | Fill ratio: how much of the residual capacity item $i$ consumes |

**Feature contract:** the feature vector produced at training time and at inference time must be identical. The serialised model bundle includes a `feature_version` integer and `n_features` count; the solver validates both on load and raises an error if either mismatches.

#### V.2.2 Model architecture

The repair ranker is a `GradientBoostingClassifier` (scikit-learn). A `StandardScaler` is fit on the training features and stored alongside the model in the bundle. At inference time, the `_FastPredictor` wrapper applies the scaler transform in pure NumPy and calls `_raw_predict` directly on the GBT, bypassing scikit-learn's per-call input validation; the output logits are converted to probabilities via sigmoid. The bin with the highest probability is selected.

Balanced sample weights are used during training to counteract class imbalance from negative subsampling. The primary evaluation metric is ROC-AUC, since the solver uses predicted probabilities for ranking rather than hard class labels.

#### V.2.3 Offline training data generation

Training rows are generated by replaying two types of traces:

- **Fresh BFD traces:** replay BFD on a fresh instance from scratch. At each insertion step where at least two bins are feasible, label BFD's chosen bin as positive and a sample of the remaining feasible bins as negative.
- **Post-destruction repair traces:** build a complete BFD solution, evict a random fraction of bins, then reinsert the displaced items into the surviving partial solution using the same labeling rule.

Pooling both trace types reduces the distribution shift between the static BFD replay states seen at training time and the mid-search partial solutions encountered during ALNS repair at inference time.
