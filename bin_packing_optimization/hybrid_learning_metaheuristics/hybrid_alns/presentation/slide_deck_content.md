---

**Slide 1**

# Hybrid ALNS for the Bin Packing Problem

### One-Dimensional Variant

A solver combining metaheuristic large-neighborhood search, adaptive bandit-based operator selection, and machine-learned repair via behavioral cloning from BFD.

---

**Slide 2** *(Introduction)*

## Introduction

**Project context:** The one-dimensional Bin-Packing Problem (1D-BPP) arises across many industrial domains — cutting stock, vehicle loading, container logistics, and cloud resource allocation — wherever discrete items must be packed into fixed-capacity bins at minimum cost.

**Why classical methods fall short:** 1D-BPP is strongly NP-hard. Exact solvers (branch-and-price) are optimal but scale to only a few hundred items. Classical metaheuristics scale well but rely on fixed decision rules that do not adapt to the problem structure encountered at runtime, leading to stagnation in dense local optima.

**Motivation for the hybrid approach:** Machine learning offers a principled way to make metaheuristic decisions adaptive — without replacing the optimization loop:
- **Online** — learn *which* destroy operator performs best given the current search context (contextual bandit)
- **Offline** — learn *which* bin to assign each displaced item by imitating an expert heuristic (behavioral cloning)

**This project** embeds both components inside an Adaptive Large-Neighborhood Search (ALNS) framework, keeping them independently togglable to enable clean ablation. The goal is to demonstrate that ML-guided decisions can meaningfully improve solution quality and adaptivity over non-learning ALNS.

---

**Slide 3**

## Outline

1. **Introduction** — project context and hybrid motivation
2. **Problem Definition** — formal model, lower bound, NP-hardness
3. **Literature Review** — ML inside metaheuristics
4. **Proposed Solution**
   - 4a. Global architecture overview
   - 4b. BFD warm start and ALNS framework
   - 4c. ML Component I — LinUCB bandit for operator selection
   - 4d. ML Component II — GBT learned repair
5. **Tests & Results** — protocol, benchmarks, ablation, multi-dataset, comparative
6. **Synthesis & Conclusion** — findings, contributions, limitations, future work

---

**Slide 4**

## I.1 Problem Definition

**Given:** item set $\mathcal{I} = \{1, \ldots, n\}$ with integer sizes $s_i \in \mathbb{Z}_{>0}$; identical bins of integer capacity $C \in \mathbb{Z}_{>0}$, with $s_i \le C$ for all $i$.

**Feasible solution** — a partition $B_1, \ldots, B_m$ of $\mathcal{I}$ satisfying:

$$\biguplus_{j=1}^{m} B_j = \mathcal{I}, \qquad \sum_{i \in B_j} s_i \le C \quad \forall\, j \in \{1,\ldots,m\}$$

**Objective:**

$$\min_{B_1,\ldots,B_m} \; m$$

---

**Slide 5**

## I.2 Continuous Relaxation Lower Bound

$$\mathrm{LB}_1 = \left\lceil \frac{\displaystyle\sum_{i=1}^{n} s_i}{C} \right\rceil$$

**Proof.** Any feasible packing must hold all item volume. With $m$ bins each contributing at most $C$ units, we need $mC \ge \sum_i s_i$, giving $m \ge \sum_i s_i / C$. Integer rounding yields the ceiling. $\square$

> Tighter bounds exist (e.g. the Martello–Toth $L_2$ bound, which accounts for large items that cannot coexist). $\mathrm{LB}_1$ is sufficient as a progress indicator in this implementation.

---

**Slide 6**

## I.3 Complexity and Algorithmic Strategy

1D-BPP is **strongly NP-hard** by reduction from 3-Partition.

| Approach      | Representative                     | Guarantee                                                 | Scalability               |
| ------------- | ---------------------------------- | --------------------------------------------------------- | ------------------------- |
| Exact         | Branch-and-bound, branch-and-price | Optimal                                                   | Up to a few hundred items |
| Approximation | FFD, BFD                           | $\le \frac{11}{9}\,\mathrm{OPT} + O(1)$, in $O(n \log n)$ | High                      |
| Metaheuristic | ALNS (this solver)                 | None (heuristic)                                          | High                      |

**Strategy:** use BFD as a deterministic warm start, then apply ALNS to escape local optima and push toward the lower bound.

---

**Slide 7** *(Literature Review)*

## II. Literature Review — ML in Metaheuristics

> **Scope:** ML *inside* metaheuristics — using learned models to guide decisions within the optimization loop. This is distinct from "metaheuristics for ML" (hyperparameter tuning, NAS), which is outside this project's scope.

**Stream 1 — Adaptive Operator Selection (AOS)**

Bandit-based AOS replaces static operator probabilities with online reward signals. Fialho et al. (2010) formalised this as a multi-armed bandit; subsequent work (COMPASS, Maturana & Saubion 2008) embedded it in evolutionary and population-based frameworks. **This project uses LinUCB (Chu et al. 2011)** — a contextual extension that conditions operator selection on search-state features, enabling fine-grained adaptation.

**Stream 2 — Learned Repair / Construction**

Khalil et al. (2017) demonstrated that graph neural networks can learn greedy construction policies competitive with classical heuristics on TSP, MVC, and MAXCUT. Behavioral cloning (imitation learning) trains a policy directly from expert demonstrations without a reward signal — applied here to replicate BFD's bin-assignment logic under partial solutions.

**Stream 3 — ML-Augmented LNS**

Hottung & Tierney (2020) and Lu et al. (2021) use deep RL to learn full destroy-and-repair operators for CVRP and VRPTW. This project takes a lighter, more interpretable approach: a GBT ranker trained offline, combined with an online bandit for operator selection.

**Positioning of this work:** at the intersection of Streams 1 and 2 — dual-learning ALNS combining online contextual bandit selection with offline supervised repair.

---

**Slide 8** *(Section Divider)*

## III. Proposed Solution

### Global Architecture · BFD Warm Start · ALNS Framework · ML Components

---

**Slide 9** *(Global Architecture Overview)*

## III.0 Global Architecture

**End-to-end pipeline of the hybrid ALNS solver:**

```
┌─────────────────────────────────────────────────────────────┐
│                    Problem Instance                          │
│              (n items, sizes sᵢ, capacity C)                 │
└────────────────────────┬────────────────────────────────────┘
                         ▼
           ┌─────────────────────────┐
           │    BFD Initialization   │  ← Constructive warm start
           │  Sort → Best-Fit Decr.  │
           └────────────┬────────────┘
                        │  x₀ (initial solution)
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                      ALNS Main Loop                          │
│                                                              │
│  ┌────────────────┐  operator  ┌────────────────────────┐   │
│  │   LinUCB       │◄───────────│    Context Vector       │   │
│  │   Bandit (ML)  │            │  [T, stagnation, gap,  │   │
│  └───────┬────────┘            │   k/n, iter progress]  │   │
│          │ select operator     └────────────────────────┘   │
│          ▼                                                   │
│  ┌────────────────┐            ┌────────────────────────┐   │
│  │  Destroy Step  │─── x̂ ─────►│   GBT Repair (ML)      │   │
│  │  (Random /     │            │  Score feasible        │   │
│  │  Worst-load /  │            │  (item, bin) pairs     │   │
│  │  Related-item) │            └───────────┬────────────┘   │
│  └────────────────┘                        │ x'              │
│                                            ▼                 │
│                          ┌─────────────────────────────┐    │
│                          │   SA Acceptance Criterion    │    │
│                          │   Accept x' or keep x        │    │
│                          │   Update x_best              │    │
│                          └──────────────┬──────────────┘    │
│                                         │ reward             │
│                                         └──────► LinUCB      │
│                                                   update      │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
                 ┌─────────────────────┐
                 │   Best Solution x*  │
                 │  gap = bins − LB₁   │
                 └─────────────────────┘
```

**Two ML decision points:** (1) LinUCB selects the destroy operator at each iteration based on search context; (2) GBT scores all feasible bin candidates during repair. Both components are independently togglable for ablation.

---

**Slide 10**

## III.1 Constructive Initialization: Best-Fit Decreasing

BFD builds a tight deterministic starting point before ALNS begins.

1. Sort items by **non-increasing size** (ties broken by item index)
2. For each item, evaluate all currently open bins that can accommodate it
3. Place the item into the feasible bin with **minimum post-placement slack**
4. If no feasible bin exists, open a new bin

> A tighter warm start leaves fewer bins for ALNS to improve from, which tends to improve final solution quality and reduces time spent recovering from a weak initial packing.

---

**Slide 11** *(Section Divider)*

## III.2 Metaheuristic Framework

### Adaptive Large-Neighborhood Search

---

**Slide 12**

## III.2a Why Local Search Alone Fails

Classical local search applies small moves within a neighborhood $\mathcal{N}(x)$ — for example, relocating a single item. While efficient, it stalls at **local optima**: solutions from which no single move improves the objective, yet which are far from globally optimal.

- Local optima in 1D-BPP are **dense**, tied to specific item groupings within bins
- No bounded sequence of single-item relocations can reliably escape them

**Solution — Large-Neighborhood Search (Shaw, 1998):**
Operate on implicitly exponential neighborhoods by partially destroying and then repairing the current solution.

---

**Slide 13**

## III.2b The LNS Iterate

Each iteration consists of two phases:

**Destroy** — remove a subset $D$ of items from their assigned bins, producing a partial solution $\hat{x}$.

**Repair** — reinsert every item in $D$ into $\hat{x}$, restoring feasibility and yielding a new complete solution $x'$.

The neighborhood is implicitly exponential in $|D|$: repair can yield any feasible completion of $\hat{x}$, allowing escape from local optima unreachable by any sequence of small moves.

ALNS (Ropke & Pisinger, 2006) extends LNS by **adaptively selecting** the destroy operator from a portfolio.

---

**Slide 14**

## III.2c Destroy Operators

Three operators are available. All guarantee that at least one item remains placed.

| Arm | Operator         | Mechanism                                                                                                                                         |
| --- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0   | **Random**       | Sample exactly $\min(k,\,n-1)$ items uniformly at random across all placed items and remove them                                                  |
| 1   | **Worst-load**   | Sort bins by ascending load (small uniform tie-breaking perturbation). Collect items in that order until $\min(k,\,n-1)$ items have been removed  |
| 2   | **Related-item** | Select a seed item uniformly at random. Remove the seed plus the $\min(k-1,\,n-2)$ items with the smallest absolute size difference from the seed |

No single operator dominates: random destruction explores broadly early on, while targeted strategies are more effective near a local optimum.

---

**Slide 15**

## III.2d Destruction Radius

The displaced-item count $k$ is drawn uniformly from $[k_{\min}, k_{\max}]$ at each iteration:

$$k_{\min} = \max\!\left(1,\,\left\lfloor 0.05\,n \right\rfloor\right), \qquad k_{\max} = \max\!\left(k_{\min}+1,\,\left\lfloor 0.25\,n \right\rfloor\right)$$

So between **5% and 25%** of items are displaced per iteration.

**Adaptive expansion.** As `iterations_since_improvement` grows toward `no_improve_limit`, $k_{\max}$ interpolates linearly upward, broadening the destruction radius to encourage diversification when the search stagnates.

---

**Slide 16** *(Section Divider)*

## III.3 Acceptance Mechanism

### Simulated Annealing with Soft Reheat and Restart

---

**Slide 17**

## III.3a SA Acceptance and Cooling

Let $\Delta = f(x') - f(x)$. Accept $x'$ if:

$$\Delta \le 0 \quad \text{or} \quad U < \exp\!\left(-\Delta/T\right), \qquad U \sim \mathrm{Uniform}(0,1)$$

**Geometric cooling** is applied after every iteration:

$$T \leftarrow \alpha_{\mathrm{cool}} \cdot T, \quad 0 < \alpha_{\mathrm{cool}} \le 1$$

**Soft reheat.** To prevent temperature collapse during prolonged stagnation, whenever `iterations_since_improvement` is a positive multiple of $\max\!\left(50,\,\lfloor\mathrm{no\_improve\_limit}/4\rfloor\right)$:

$$T \leftarrow \max(T,\;0.35\cdot T_0)$$

---

**Slide 18**

## III.3b Diversification Restart

When `iterations_since_improvement` reaches `no_improve_limit`:

1. The current solution is reset to the **best solution found so far**
2. Temperature is **reheated**: $T \leftarrow \max(T,\;0.20\cdot T_0)$
3. The patience window **shrinks**:

$$\mathrm{no\_improve\_limit} \leftarrow \max\!\left(100,\,\left\lfloor\tfrac{2}{3}\,\mathrm{no\_improve\_limit}\right\rfloor\right)$$

so successive restarts trigger progressively sooner

4. `iterations_since_improvement` is reset to zero

The outer budget `max_iterations` is the **sole hard termination criterion**; the restart mechanism never terminates the run.

---

**Slide 19** *(Section Divider)*

## IV. Machine Learning Components

### Operator Selection + Learned Repair

---

**Slide 20**

## IV. Two Independent ML Components

| Component             | Decision Targeted                              | Method                   |
| --------------------- | ---------------------------------------------- | ------------------------ |
| **Contextual Bandit** | Which destroy operator to apply each iteration | Warm-start LinUCB        |
| **Repair Ranker**     | Which bin to assign each displaced item        | Behavioral cloning (GBT) |

The two components are **independent** in implementation and can be enabled or disabled separately. Component I selects the operator used during destruction; Component II guides the subsequent repair.

---

**Slide 21**

## IV.1 Operator Selection: Phase 1

### Warm-up — Beta-Bernoulli Thompson Sampling (first 300 calls)

All arms initialised at $\mathrm{Beta}(1,1)$ (uniform prior).

For each call:

1. Sample $\tilde{\theta}_k \sim \mathrm{Beta}(\alpha_k,\beta_k)$ for each arm $k \in \{0,1,2\}$
2. Select $k^* = \arg\max_k \tilde{\theta}_k$
3. Observe reward $r \in [0,1]$; draw $\tilde{r} \sim \mathrm{Bernoulli}(r)$, then update $k^*$ only:
   - $\tilde{r} = 1 \;\Rightarrow\; \alpha_{k^*} \leftarrow \alpha_{k^*}+1$
   - $\tilde{r} = 0 \;\Rightarrow\; \beta_{k^*} \leftarrow \beta_{k^*}+1$

This phase identifies well-performing operators without requiring the context vector, mitigating the cold-start problem of pure LinUCB on short runs.

---

**Slide 22**

## IV.1 Operator Selection: Phase 2

### Exploitation — Disjoint LinUCB (calls 301+, $\alpha = 0.3$, Chu et al. 2011)

Each arm $k$ maintains $A_k^{-1} \in \mathbb{R}^{5\times5}$ and $\mathbf{b}_k \in \mathbb{R}^5$.

**Selection.** Given context $\mathbf{x} \in \mathbb{R}^5$:

$$k^* = \arg\max_k \Bigl[\hat{\boldsymbol{\theta}}_k^\top\mathbf{x} + \alpha\sqrt{\mathbf{x}^\top A_k^{-1}\mathbf{x}}\Bigr], \qquad \hat{\boldsymbol{\theta}}_k = A_k^{-1}\mathbf{b}_k$$

**Update** via Sherman-Morrison rank-1 formula ($O(d^2)$):

$$A_{k^*}^{-1} \leftarrow A_{k^*}^{-1} - \frac{(A_{k^*}^{-1}\mathbf{x})(A_{k^*}^{-1}\mathbf{x})^\top}{1+\mathbf{x}^\top A_{k^*}^{-1}\mathbf{x}}, \qquad \mathbf{b}_{k^*} \leftarrow \mathbf{b}_{k^*} + r\,\mathbf{x}$$

---

**Slide 23**

## IV.1 Context Vector

The 5-dimensional feature $\mathbf{x}$ encodes the current search state:

| Index | Feature                                | Description                                       |
| ----- | -------------------------------------- | ------------------------------------------------- |
| 0     | $T / T_0$                              | Normalised temperature; 1 = start (hot), 0 = cold |
| 1     | `iter_no_improve` / `no_improve_limit` | Stagnation progress toward the next restart       |
| 2     | $\mathrm{LB}_1 / f(x)$                 | Lower-bound-to-cost ratio; 1.0 = optimal          |
| 3     | $k / n$                                | Current destruction radius as a fraction of $n$   |
| 4     | `iteration` / `max_iterations`         | Overall search progress (0 to 1)                  |

All features lie in $[0,1]$, enabling the UCB exploration term to be comparable across dimensions without additional normalisation.

---

**Slide 24**

## IV.1 Reward Signal

Let $\mathrm{gap} = \max(1,\,f(x) - \mathrm{LB}_1)$ be the current gap after updating the incumbent.

$$r = \begin{cases} \min\!\left(1,\;\dfrac{\max(0,-\Delta)}{\mathrm{gap}}\right) & \text{bins were saved } (\Delta < 0) \\[8pt] 0.2 & \text{accepted without saving bins} \\[6pt] 0.0 & \text{rejected} \end{cases}$$

> Normalising by $\mathrm{gap}$ rewards bin savings more highly when the solution is already close to the lower bound, reflecting the increasing marginal difficulty of further improvement.

---

**Slide 25**

## IV.2 Machine-Learned Repair: Overview

After destruction, each displaced item $i$ must be reinserted. Let $\mathcal{F}(i) = \{j : \ell_j + s_i \le C\}$ be the set of feasible bins.

- $\mathcal{F}(i) = \emptyset$: open a new bin immediately
- Otherwise: **score every bin** in $\mathcal{F}(i)$; place item $i$ in the **highest-scoring bin**
- Items processed in **non-increasing size order**

**Training objective:** binary classifier on feasible $(\text{item},\text{bin})$ pairs. Label 1 if BFD would have chosen that bin, label 0 otherwise. This is behavioral cloning from the BFD expert.

---

**Slide 26**

## IV.2 Feature Representation

Each feasible $(i,j)$ pair is encoded as an **11-dimensional vector** (all features normalised by $C$ or $n$):

| Idx | Feature                    | Description                                                    |
| --- | -------------------------- | -------------------------------------------------------------- |
| 0   | $s_i / C$                  | Normalised item size                                           |
| 1   | $(s_i / C)^2$              | Squared normalised size                                        |
| 2   | $\mathrm{rank}(i) / n$     | Size rank among $n$ items (0 = largest)                        |
| 3   | $\mathrm{remaining} / n$   | Fraction of displaced items not yet reinserted                 |
| 4   | $\ell_j / C$               | Normalised current bin load                                    |
| 5   | $(C - \ell_j) / C$         | Residual capacity fraction                                     |
| 6   | $(C - \ell_j - s_i) / C$   | Post-placement slack fraction                                  |
| 7   | $\lvert B_j \rvert / n$    | Normalised bin occupancy (item count)                          |
| 8   | $\max_{k \in B_j} s_k / C$ | Largest item already in bin $j$                                |
| 9   | $\min_{k \in B_j} s_k / C$ | Smallest item already in bin $j$                               |
| 10  | $s_i / (C - \ell_j)$       | Fill ratio: fraction of residual capacity consumed by item $i$ |

---

**Slide 27**

## IV.2 Model Architecture and Training Data

**Repair ranker:** `GradientBoostingClassifier` (scikit-learn)

- `StandardScaler` fit on training features, serialised with the model
- Inference via `_FastPredictor`: scaler applied in NumPy, `_raw_predict` called directly on the GBT — bypasses per-call validation; logits converted to probabilities via sigmoid
- Bundle stores `feature_version` and `n_features`; mismatch raises an error at load time
- Balanced sample weights; primary metric: **ROC-AUC** (probabilities used for bin ranking, not hard class labels)

**Training data** — two trace types pooled to reduce distribution shift:

- **Fresh BFD traces:** replay BFD from scratch; label chosen bin positive, sample remaining feasible bins negative
- **Post-destruction traces:** build a BFD solution, evict a random fraction of bins, reinsert displaced items with the same labeling rule

---

**Slide 28** *(Section Divider)*

## V. Experimental Evaluation

### Protocol · Benchmarks · Ablation · Results

---

**Slide 29**

## V.1 Experimental Protocol

| Setting                                      | Value                                    |
| -------------------------------------------- | ---------------------------------------- |
| Random seed                                  | 42                                       |
| ALNS iterations                              | 300                                      |
| Initial temperature $T_0$                    | $1/\ln 2$                                |
| Cooling coefficient $\alpha_{\mathrm{cool}}$ | 0.9995                                   |
| LinUCB exploration $\alpha$                  | 0.3                                      |
| Thompson-sampling warm-up                    | 300 calls                                |
| Repair model                                 | Single pre-trained bundle (all datasets) |

**Primary metric throughout:**

$$\text{gap} = \text{bins used} - \mathrm{LB}_1 \qquad (\text{gap} = 0 \;\Leftrightarrow\; \text{LB-certifiably optimal})$$

**Environment:** Python 3.13 · scikit-learn · Windows 11 · Intel i9-13950HX · 64 GB RAM

---

**Slide 30**

## V.2 Benchmark Datasets — Instance Characteristics

Five families covering structurally distinct regimes:

| Family       | Capacity $C$ | Items $n$ | Structure          |
| ------------ | ------------ | --------- | ------------------ |
| Scholl-2     | 1 000        | 50 – 500  | Uniform sizes      |
| Falkenauer-T | 1 000        | —         | Triplet structure  |
| Falkenauer-U | 150          | —         | Uniform sizes      |
| Wäscher      | 10 000       | —         | Cutting-stock      |
| Hard28       | 1 000        | 160 – 200 | Adversarially hard |

> **Falkenauer-T note:** the triplet structure systematically weakens $\mathrm{LB}_1$ relative to the true optimum. Reported gaps for this family are **not directly comparable** to those of other families.

---

**Slide 31**

## V.3 Test 1 — Ablation Study

**Purpose of this test:** Isolate the individual contribution of each ML component (LinUCB bandit and GBT repair ranker) by toggling each independently. This determines which component drives quality improvement and which adds computational overhead without proportional benefit, validating the design choices of the hybrid architecture.

**Scholl-2 · 5 instances · 50 items** — each component toggled independently.

| Configuration          | Avg. bins | Avg. gap | Fill %    | Time (s)  |
| ---------------------- | --------- | -------- | --------- | --------- |
| No learning (baseline) | 18.2      | 0.20     | 92.76     | 0.153     |
| **Online RL only**     | **18.0**  | **0.00** | **93.76** | **0.186** |
| Offline GBT only       | 18.2      | 0.20     | 92.76     | 0.353     |
| Both combined          | 18.2      | 0.20     | 92.76     | 0.362     |

**Key finding:** the online RL selector is the **decisive contributor** — it closes the gap to 0 alone. The offline GBT repair model adds approximately **2× runtime** with no quality gain on this small-instance slice; its benefit is expected to emerge at larger scale.

> Conclusions are bounded by a 5-instance evaluation slice at a single instance size. Large-instance behaviour is an open empirical question.

---

**Slide 32**

## V.4 Test 2 — Multi-Dataset Generalization

**Purpose of this test:** Evaluate the generalization of the combined method (online RL + offline GBT) across structurally distinct benchmark families. Assesses whether the hybrid approach adapts to different item size distributions, capacity regimes, and structural properties — including adversarially hard instances — without retuning any parameter.

**Combined method** (online RL + offline GBT repair) across all benchmark families.

| Family       | Mean gap | Runtime (s) |
| ------------ | -------- | ----------- |
| Scholl-2     | 0.20     | 0.6 – 1.2   |
| Falkenauer-T | 1.00     | 1.0 – 1.7   |
| Falkenauer-U | 0.20     | 4.3 – 5.8   |
| Wäscher      | 0.60     | 1.0 – 2.2   |
| Hard28       | 0.67     | 13.5 – 15.1 |

**4 of 5 families** land within 1 bin of $\mathrm{LB}_1$. The Falkenauer-T gap of 1.00 most likely reflects the weakness of $\mathrm{LB}_1$ on triplet instances rather than degraded search quality.

---

**Slide 33**

## V.5 Test 3 — Comparative Study

**Purpose of this test:** Benchmark the dual-learning ALNS against a representative set of classical combinatorial optimization methods — constructive, local search, evolutionary, and population-based — on a common instance set. Assesses both solution quality (gap to $\mathrm{LB}_1$) and computational cost (runtime) to determine where the hybrid approach stands in the quality–speed trade-off landscape.

**Scholl-2 · 5 instances · 50 items** — single-run comparison.

| Method                   | Mean gap | Time (s) |
| ------------------------ | -------- | -------- |
| FFD / BFD (constructive) | 1.80     | < 0.01   |
| Simulated Annealing      | 1.80     | 0.21     |
| Tabu Search              | 1.80     | 0.13     |
| Genetic Algorithm        | 1.00     | 0.82     |
| **Dual-learning ALNS**   | **0.20** | **0.36** |
| Ant Colony Optimization  | 0.00     | 3.03     |

The dual-learning ALNS achieves a mean gap of **0.20** — the lowest of any tested method except ACO — while running **8× faster** than ACO. No other method in the comparison achieves both a lower gap and a lower runtime.

> Stochastic baselines are single-run results; multi-seed averaging may alter relative rankings.

---

**Slide 34** *(Section Divider)*

## VI. Synthesis & Conclusion

### Contributions · Limitations · Future Work

---

**Slide 35**

## VI.1 Synthesis of Results

| Finding                                               | Evidence                                                        |
| ----------------------------------------------------- | --------------------------------------------------------------- |
| Online RL selector is the decisive ML contributor     | Online-only closes gap 0.20 → 0.00; combined does not           |
| Offline GBT repair: implemented, not yet impactful    | No quality gain; +2× runtime on small instances                 |
| Combined method generalises across families           | ≤ 1 bin from $\mathrm{LB}_1$ on 4 / 5 families                  |
| The two components contribute asymmetrically          | Ablation reveals clear imbalance — clear avenue for improvement |
| Dual-learning ALNS dominates classical metaheuristics | Best gap–runtime trade-off in comparative study                 |

> All conclusions are bounded by **5-instance evaluation slices** and single-run stochastic baselines. Broader empirical validation is required before general claims can be made.

---

**Slide 36**

## VI.2 Conclusions

**Three contributions:**

1. **Dual-learning ALNS for 1D-BPP** — online and offline ML components with independent toggle switches enabling clean, separable ablation.
2. **Supervised repair pipeline** — DAgger-lite data augmentation pooling fresh-BFD and post-destruction traces, 11-feature v2 contract, strict quality gates (ROC-AUC, avg. precision), and serialised feature versioning to prevent silent mismatch.
3. **Warm-start LinUCB bandit** — Beta–Bernoulli Thompson Sampling cold-start mitigation transitioning to disjoint LinUCB after 300 calls, with a gap-normalised reward signal that scales difficulty appropriately.

**Limitations:** small evaluation slices · single-run stochastic baselines · $\mathrm{LB}_1$ only (weaker than Martello–Toth $L_2$) · GBT overhead not recovered at small scale.

**Future directions:** full benchmark evaluation with averaged runs · upgrade to $L_2$ lower bound · end-to-end RL repair to remove the supervised dependency · cross-family transfer of learned components.
