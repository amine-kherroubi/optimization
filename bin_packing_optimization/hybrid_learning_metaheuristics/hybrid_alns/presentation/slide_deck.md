---
marp: true
theme: default
math: mathjax
size: 16:9
paginate: true
style: |
  @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;0,700;1,500&family=Jost:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {
    --navy:   #14273f;
    --gold:   #b8913a;
    --bg:     #f7f5f1;
    --text:   #1b1b1e;
    --muted:  #5a6b80;
    --rule:   #ddd5c5;
    --callout:#eee8dc;
    --codbg:  #e8e2d8;
  }

  section {
    font-family: 'Jost', sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 44px 64px 44px 64px;
    font-size: 21px;
    line-height: 1.55;
    font-weight: 300;
    letter-spacing: 0.01em;
  }

  /* ── Headings ── */
  h1 {
    font-family: 'Cormorant Garamond', serif;
    color: var(--navy);
    font-size: 1.95em;
    font-weight: 700;
    line-height: 1.2;
    border-bottom: 2px solid var(--gold);
    padding-bottom: 0.25em;
    margin-bottom: 0.55em;
    margin-top: 0;
    letter-spacing: -0.01em;
  }

  h2 {
    font-family: 'Cormorant Garamond', serif;
    color: var(--navy);
    font-size: 1.55em;
    font-weight: 600;
    margin: 0.25em 0 0.4em;
    line-height: 1.25;
  }

  h3 {
    font-family: 'Jost', sans-serif;
    color: var(--gold);
    font-size: 0.72em;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin: 0.6em 0 0.35em;
  }

  /* ── Body text ── */
  strong {
    color: var(--navy);
    font-weight: 600;
  }

  p { margin: 0.35em 0; }

  ul, ol {
    margin: 0.3em 0 0.3em 1.4em;
    padding: 0;
  }

  li {
    margin-bottom: 0.25em;
    padding-left: 0.15em;
  }

  /* ── Tables ── */
  section table {
    width: 100% !important;
    min-width: 100% !important;
    border-collapse: collapse;
    font-size: 0.8em;
    margin-top: 0.5em;
    line-height: 1.4;
  }

  th {
    background: var(--navy);
    color: #dce8f4;
    padding: 7px 13px;
    text-align: left;
    font-family: 'Jost', sans-serif;
    font-weight: 500;
    letter-spacing: 0.04em;
    font-size: 0.9em;
  }

  td {
    padding: 6px 13px;
    border-bottom: 1px solid var(--rule);
    vertical-align: top;
    background: transparent;
  }

  /* ── Code ── */
  code {
    font-family: 'JetBrains Mono', monospace;
    background: var(--codbg);
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 0.82em;
    font-weight: 400;
  }

  /* ── Blockquote as callout ── */
  blockquote {
    border-left: 3px solid var(--gold);
    background: var(--callout);
    margin: 0.6em 0 0.2em;
    padding: 0.45em 1em;
    border-radius: 0 4px 4px 0;
    font-style: normal;
    font-size: 0.9em;
    color: #3a3020;
  }

  blockquote p { margin: 0; }

  /* ── Pagination ── */
  section::after {
    font-family: 'Jost', sans-serif;
    font-size: 0.62em;
    color: var(--muted);
  }

  /* ── Title slide ── */
  section.lead {
    background: var(--navy);
    color: #f0f4f8;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 60px 80px;
  }

  section.lead::before {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 5px;
    background: var(--gold);
  }

  section.lead h1 {
    font-size: 2.6em;
    color: #ffffff;
    border-bottom: 2px solid var(--gold);
    padding-bottom: 0.3em;
    margin-bottom: 0.4em;
    font-weight: 700;
  }

  section.lead h3 {
    color: #8bacc8;
    font-size: 0.75em;
    letter-spacing: 0.16em;
    margin-bottom: 0.8em;
  }

  section.lead p {
    color: #8bacc8;
    font-size: 0.88em;
    font-weight: 300;
    max-width: 78%;
    line-height: 1.6;
  }

  section.lead::after { color: #3a5570; }

  /* ── Section divider slides ── */
  section.divider {
    background: var(--navy);
    color: #f0f4f8;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 60px 80px;
  }

  section.divider::before {
    content: '';
    position: absolute;
    left: 80px;
    top: 50%;
    transform: translateY(-50%);
    width: 4px;
    height: 40%;
    background: var(--gold);
  }

  section.divider h2 {
    font-size: 2.4em;
    color: #ffffff;
    font-weight: 700;
    margin-left: 28px;
    line-height: 1.15;
  }

  section.divider h3 {
    color: #8bacc8;
    margin-left: 28px;
    font-size: 0.74em;
  }

  section.divider::after { color: #3a5570; }

  /* ── Compact slides (dense tables) ── */
  section.compact {
    font-size: 17.5px;
  }

  section.compact table { font-size: 0.82em; }
  section.compact td, section.compact th { padding: 5px 10px; }
---

<!-- _class: lead -->
<!-- _paginate: false -->

# Hybrid ALNS for the<br>Bin Packing Problem

### One-Dimensional Variant

A solver combining metaheuristic large-neighborhood search, adaptive bandit-based operator selection, and machine-learned repair via behavioral cloning from BFD.

---

## Outline

1. **Problem Definition** — formal model, lower bound, complexity
2. **BFD Initialization** — constructive warm start
3. **ALNS Framework** — destroy operators, destruction radius
4. **SA Acceptance** — cooling schedule, soft reheat, restart
5. **ML Component I** — contextual bandit (LinUCB) for operator selection
6. **ML Component II** — learned repair via behavioral cloning
7. **Experimental Evaluation** — protocol, benchmarks, ablation, results
8. **Conclusions** — synthesis, contributions, limitations, future work

---

## I. Problem Definition

**Given:** item set $\mathcal{I} = \{1, \ldots, n\}$ with integer sizes $s_i \in \mathbb{Z}_{>0}$; identical bins of integer capacity $C \in \mathbb{Z}_{>0}$, with $s_i \le C$ for all $i$.

**Feasible solution** — a partition $B_1, \ldots, B_m$ of $\mathcal{I}$ satisfying:

$$\biguplus_{j=1}^{m} B_j = \mathcal{I}, \qquad \sum_{i \in B_j} s_i \le C \quad \forall\, j \in \{1,\ldots,m\}$$

**Objective:**

$$\min_{B_1,\ldots,B_m} \; m$$

---

## I.2 Continuous Relaxation Lower Bound

$$\mathrm{LB}_1 = \left\lceil \frac{\displaystyle\sum_{i=1}^{n} s_i}{C} \right\rceil$$

**Proof.** Any feasible packing must hold all item volume. With $m$ bins each contributing at most $C$ units, we need $mC \ge \sum_i s_i$, giving $m \ge \sum_i s_i / C$. Integer rounding yields the ceiling. $\square$

> Tighter bounds exist (e.g. the Martello-Toth $L_2$ bound, which accounts for large items that cannot coexist). $\mathrm{LB}_1$ is sufficient as a progress indicator in this implementation.

---

## I.3 Complexity and Algorithmic Strategy

1D-BPP is **strongly NP-hard** by reduction from 3-Partition.

| Approach      | Representative                     | Guarantee                                                 | Scalability               |
| ------------- | ---------------------------------- | --------------------------------------------------------- | ------------------------- |
| Exact         | Branch-and-bound, branch-and-price | Optimal                                                   | Up to a few hundred items |
| Approximation | FFD, BFD                           | $\le \frac{11}{9}\,\mathrm{OPT} + O(1)$, in $O(n \log n)$ | High                      |
| Metaheuristic | ALNS (this solver)                 | None (heuristic)                                          | High                      |

**Strategy:** use BFD as a deterministic warm start, then apply ALNS to escape local optima and push toward the lower bound.

---

## II. Constructive Initialization: Best-Fit Decreasing

BFD builds a tight deterministic starting point before ALNS begins.

1. Sort items by **non-increasing size** (ties broken by item index)
2. For each item, evaluate all currently open bins that can accommodate it
3. Place the item into the feasible bin with **minimum post-placement slack**
4. If no feasible bin exists, open a new bin

> A tighter warm start leaves fewer bins for ALNS to improve from, which tends to improve final solution quality and reduces time spent recovering from a weak initial packing.

---

<!-- _class: divider -->
<!-- _paginate: false -->

## III. Metaheuristic Framework

### Adaptive Large-Neighborhood Search

---

## III.1 Why Local Search Alone Fails

Classical local search applies small moves within a neighborhood $\mathcal{N}(x)$ — for example, relocating a single item. While efficient, it stalls at **local optima**: solutions from which no single move improves the objective, yet which are far from globally optimal.

- Local optima in 1D-BPP are **dense**, tied to specific item groupings within bins
- No bounded sequence of single-item relocations can reliably escape them

**Solution — Large-Neighborhood Search (Shaw, 1998):**
operate on implicitly exponential neighborhoods by partially destroying and then repairing the current solution.

---

## III.2 The LNS Iterate

Each iteration consists of two phases:

**Destroy** — remove a subset $D$ of items from their assigned bins, producing a partial solution $\hat{x}$ and a displaced item set $D$.

**Repair** — reinsert every item in $D$ into $\hat{x}$, restoring feasibility and yielding a new complete solution $x'$.

The neighborhood is implicitly exponential in $|D|$: repair can yield any feasible completion of $\hat{x}$, allowing escape from local optima unreachable by any sequence of small moves.

ALNS (Ropke & Pisinger, 2006) extends LNS by **adaptively selecting** the destroy operator from a portfolio.

---

## III.3 Destroy Operators

<!-- _class: compact -->

Three operators are available. All guarantee that at least one item remains placed.

| Arm | Operator         | Mechanism                                                                                                                                         |
| --- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0   | **Random**       | Sample exactly $\min(k,\,n-1)$ items uniformly at random across all placed items and remove them                                                  |
| 1   | **Worst-load**   | Sort bins by ascending load (small uniform tie-breaking perturbation). Collect items in that order until $\min(k,\,n-1)$ items have been removed  |
| 2   | **Related-item** | Select a seed item uniformly at random. Remove the seed plus the $\min(k-1,\,n-2)$ items with the smallest absolute size difference from the seed |

No single operator dominates: random destruction explores broadly early on, while targeted strategies are more effective near a local optimum.

---

## III.4 Destruction Radius

The displaced-item count $k$ is drawn uniformly from $[k_{\min}, k_{\max}]$ at each iteration:

$$k_{\min} = \max\!\left(1,\,\left\lfloor 0.05\,n \right\rfloor\right), \qquad k_{\max} = \max\!\left(k_{\min}+1,\,\left\lfloor 0.25\,n \right\rfloor\right)$$

So between **5% and 25%** of items are displaced per iteration.

**Adaptive expansion.** As `iterations_since_improvement` grows toward `no_improve_limit`, $k_{\max}$ interpolates linearly upward, broadening the destruction radius to encourage diversification when the search stagnates.

---

<!-- _class: divider -->
<!-- _paginate: false -->

## IV. Acceptance Mechanism

### Simulated Annealing with Soft Reheat and Restart

---

## IV.1 SA Acceptance and Cooling

Let $\Delta = f(x') - f(x)$. Accept $x'$ if:

$$\Delta \le 0 \quad \text{or} \quad U < \exp\!\left(-\Delta/T\right), \qquad U \sim \mathrm{Uniform}(0,1)$$

**Geometric cooling** is applied after every iteration:

$$T \leftarrow \alpha_{\mathrm{cool}} \cdot T, \quad 0 < \alpha_{\mathrm{cool}} \le 1$$

**Soft reheat.** To prevent temperature collapse during prolonged stagnation, whenever `iterations_since_improvement` is a positive multiple of $\max\!\left(50,\,\lfloor\mathrm{no\_improve\_limit}/4\rfloor\right)$:

$$T \leftarrow \max(T,\;0.35\cdot T_0)$$

---

## IV.2 Diversification Restart

When `iterations_since_improvement` reaches `no_improve_limit`:

1. The current solution is reset to the **best solution found so far**
2. Temperature is **reheated**: $T \leftarrow \max(T,\;0.20\cdot T_0)$
3. The patience window **shrinks**:
$$\mathrm{no\_improve\_limit} \leftarrow \max\!\left(100,\,\left\lfloor\tfrac{2}{3}\,\mathrm{no\_improve\_limit}\right\rfloor\right)$$
so successive restarts trigger progressively sooner
4. `iterations_since_improvement` is reset to zero

The outer budget `max_iterations` is the **sole hard termination criterion**; the restart mechanism never terminates the run.

---

<!-- _class: divider -->
<!-- _paginate: false -->

## V. Machine Learning Components

### Operator Selection + Learned Repair

---

## V. Two Independent ML Components

| Component             | Decision Targeted                              | Method                   |
| --------------------- | ---------------------------------------------- | ------------------------ |
| **Contextual Bandit** | Which destroy operator to apply each iteration | Warm-start LinUCB        |
| **Repair Ranker**     | Which bin to assign each displaced item        | Behavioral cloning (GBT) |

The two components are **independent** in implementation and can be enabled or disabled separately. Component I selects the operator used during destruction; Component II guides the subsequent repair.

---

## V.1 Operator Selection: Phase 1

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

## V.1 Operator Selection: Phase 2

### Exploitation — Disjoint LinUCB (calls 301+, $\alpha = 0.3$, Chu et al. 2011)

Each arm $k$ maintains $A_k^{-1} \in \mathbb{R}^{5\times5}$ and $\mathbf{b}_k \in \mathbb{R}^5$.

**Selection.** Given context $\mathbf{x} \in \mathbb{R}^5$:

$$k^* = \arg\max_k \Bigl[\hat{\boldsymbol{\theta}}_k^\top\mathbf{x} + \alpha\sqrt{\mathbf{x}^\top A_k^{-1}\mathbf{x}}\Bigr], \qquad \hat{\boldsymbol{\theta}}_k = A_k^{-1}\mathbf{b}_k$$

**Update** via Sherman-Morrison rank-1 formula ($O(d^2)$):

$$A_{k^*}^{-1} \leftarrow A_{k^*}^{-1} - \frac{(A_{k^*}^{-1}\mathbf{x})(A_{k^*}^{-1}\mathbf{x})^\top}{1+\mathbf{x}^\top A_{k^*}^{-1}\mathbf{x}}, \qquad \mathbf{b}_{k^*} \leftarrow \mathbf{b}_{k^*} + r\,\mathbf{x}$$

---

## V.1 Context Vector

<!-- _class: compact -->

The 5-dimensional feature $\mathbf{x}$ encodes the current search state:

| Index | Feature                                | Description                                         |
| ----- | -------------------------------------- | --------------------------------------------------- |
| 0     | $T / T_0$                              | Normalised temperature; 1 = start (hot), 0 = cold   |
| 1     | `iter_no_improve` / `no_improve_limit` | Stagnation progress toward the next restart         |
| 2     | $f(x) / \mathrm{LB}_1$                 | Ratio of current cost to lower bound; 1.0 = optimal |
| 3     | $k / n$                                | Current destruction radius as a fraction of $n$     |
| 4     | `iteration` / `max_iterations`         | Overall search progress (0 to 1)                    |

All features lie in $[0,1]$, enabling the UCB exploration term to be comparable across dimensions without additional normalisation.

---

## V.1 Reward Signal

Let $\mathrm{gap} = \max(1,\,f(x) - \mathrm{LB}_1)$ be the current gap after updating the incumbent.

$$r = \begin{cases} \min\!\left(1,\;\dfrac{\max(0,-\Delta)}{\mathrm{gap}}\right) & \text{bins were saved } (\Delta < 0) \\[8pt] 0.2 & \text{accepted without saving bins} \\[6pt] 0.0 & \text{rejected} \end{cases}$$

> Normalising by $\mathrm{gap}$ rewards bin savings more highly when the solution is already close to the lower bound, reflecting the increasing marginal difficulty of further improvement.

---

## V.2 Machine-Learned Repair: Overview

After destruction, each displaced item $i$ must be reinserted. Let $\mathcal{F}(i) = \{j : \ell_j + s_i \le C\}$ be the set of feasible bins.

- $\mathcal{F}(i) = \emptyset$: open a new bin immediately
- Otherwise: **score every bin** in $\mathcal{F}(i)$; place item $i$ in the **highest-scoring bin**
- Items processed in **non-increasing size order**

**Training objective:** binary classifier on feasible $(\text{item},\text{bin})$ pairs. Label 1 if BFD would have chosen that bin, label 0 otherwise. This is behavioral cloning from the BFD expert.

---

## V.2 Feature Representation

<!-- _class: compact -->

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

## V.2 Model Architecture and Training Data

**Repair ranker:** `GradientBoostingClassifier` (scikit-learn)

- `StandardScaler` fit on training features, serialised with the model
- Inference via `_FastPredictor`: scaler applied in NumPy, `_raw_predict` called directly on the GBT — bypasses per-call validation; logits converted to probabilities via sigmoid
- Bundle stores `feature_version` and `n_features`; mismatch raises an error at load time
- Balanced sample weights; primary metric: **ROC-AUC** (probabilities used for bin ranking, not hard class labels)

**Training data** — two trace types pooled to reduce distribution shift:

- **Fresh BFD traces:** replay BFD from scratch; label chosen bin positive, sample remaining feasible bins negative
- **Post-destruction traces:** build a BFD solution, evict a random fraction of bins, reinsert displaced items with the same labeling rule

---

<!-- _class: divider -->
<!-- _paginate: false -->

## VI. Experimental Evaluation

### Protocol · Benchmarks · Ablation · Results

---

## VI.1 Experimental Protocol

<!-- _class: compact -->

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

## VI.2 Benchmark Datasets

<!-- _class: compact -->

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

## VI.3 Ablation Study

<!-- _class: compact -->

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

## VI.4 Multi-Dataset Results

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

## VI.5 Comparative Study

<!-- _class: compact -->

**Scholl-2 · 5 instances · 50 items** — single-run comparison against standard combinatorial-optimization methods.

| Method                   | Mean gap | Time (s) |
| ------------------------ | -------- | -------- |
| FFD / BFD (constructive) | 1.80     | < 0.01   |
| Simulated Annealing      | 1.80     | 0.21     |
| Tabu Search              | 1.80     | 0.13     |
| Genetic Algorithm        | 1.00     | 0.82     |
| **Dual-learning ALNS**   | **0.20** | **0.36** |
| Ant Colony Optimization  | 0.00     | 3.03     |

The dual-learning ALNS reaches near-optimal quality at **8× lower runtime** than ACO, dominating every other metaheuristic tested on both dimensions.

> Stochastic baselines are single-run results; multi-seed averaging may alter relative rankings.

---

<!-- _class: divider -->
<!-- _paginate: false -->

## VII. Conclusions

### Contributions · Limitations · Future Work

---

## VII.1 Synthesis

| Finding                                               | Evidence                                                        |
| ----------------------------------------------------- | --------------------------------------------------------------- |
| Online RL selector is the decisive ML contributor     | Online-only closes gap 0.20 → 0.00; combined does not           |
| Offline GBT repair: implemented, not yet impactful    | No quality gain; +2× runtime on small instances                 |
| Combined method generalises across families           | ≤ 1 bin from $\mathrm{LB}_1$ on 4 / 5 families                  |
| The two components contribute asymmetrically          | Ablation reveals clear imbalance — clear avenue for improvement |
| Dual-learning ALNS dominates classical metaheuristics | Best gap–runtime trade-off in comparative study                 |

> All conclusions are bounded by **5-instance evaluation slices** and single-run stochastic baselines. Broader empirical validation is required before general claims can be made.

---

## VII.2 Conclusion

**Three contributions:**

1. **Dual-learning ALNS for 1D-BPP** — online and offline ML components with independent toggle switches enabling clean, separable ablation.
2. **Supervised repair pipeline** — DAgger-lite data augmentation pooling fresh-BFD and post-destruction traces, 11-feature v2 contract, strict quality gates (ROC-AUC, avg. precision), and serialised feature versioning to prevent silent mismatch.
3. **Warm-start LinUCB bandit** — Beta–Bernoulli Thompson Sampling cold-start mitigation transitioning to disjoint LinUCB after 300 calls, with a gap-normalised reward signal that scales difficulty appropriately.

**Limitations:** small evaluation slices · single-run stochastic baselines · $\mathrm{LB}_1$ only (weaker than Martello–Toth $L_2$) · GBT overhead not recovered at small scale.

**Future directions:** full benchmark evaluation with averaged runs · upgrade to $L_2$ lower bound · end-to-end RL repair to remove the supervised dependency · cross-family transfer of learned components.
