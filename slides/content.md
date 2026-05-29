---
marp: true
---

# Introduction: why this project matters

- The repository studies the one-dimensional Bin Packing Problem, where industrial logistics, cutting-stock, scheduling, and cloud-allocation decisions are abstracted as packing positive-size items into the fewest identical bins.
- The implemented solver treats the problem as a large-scale combinatorial optimization task where exact methods are not the practical focus, and the active implementation is a hybrid ALNS pipeline under `hybrid_learning_metaheuristics/hybrid_alns`.
- The core project idea is to keep a classical destroy--repair ALNS backbone while adding two learning decisions: an offline learned repair model and an online RL-style destroy-operator selector.
- Artifact anchor: `article/main.tex`, `approach_explanation.md`, and `hybrid_alns_solver.py` jointly define the paper narrative, mathematical framing, and executable solver.

---

# Introduction: dual-learning ALNS in one sentence

- The implemented system begins from a Best-Fit Decreasing warm start, repeatedly destroys and repairs the incumbent solution, accepts candidates through simulated annealing, and returns the best packing found.
- The offline component is a gradient-boosted classifier trained to rank feasible bins during repair, using capacity-normalized `(item, bin)` features and BFD-derived labels.
- The online component is a warm-start LinUCB contextual bandit that selects one of three destroy operators from a five-dimensional search context.
- The repository evidence shows that these two ML components are separable switches in the solver, which enables the ablation study reported in the article.

---

# Problem Definition: 1D Bin Packing Problem

- The input is a finite item set \(\mathcal{I}=\{1,\dots,n\}\), strictly positive item sizes \(s_i\), and identical bins of capacity \(C\) with the feasibility assumption \(s_i\le C\).
- A feasible solution partitions all items into non-empty bins such that the total size assigned to every bin is at most \(C\).
- The objective is to minimize the number of bins used, which is the value returned by the solver as `total_bins_used`.
- The repository uses the continuous lower bound \(\mathrm{LB}_1=\lceil\sum_i s_i/C\rceil\) as the progress and gap reference in evaluation.

---

# Problem Formulation: ILP used in the article

- The article states binary assignment variables \(x_{ij}=1\) when item \(i\) is placed in bin \(j\), and binary usage variables \(y_j=1\) when bin \(j\) is opened.
- The objective is \(\min \sum_{j=1}^{n} y_j\), which directly minimizes the number of opened bins.
- The assignment constraints \(\sum_{j=1}^{n}x_{ij}=1\) force every item to appear in exactly one bin.
- The capacity-linking constraints \(\sum_{i=1}^{n}s_i x_{ij}\le C y_j\) and \(x_{ij}\le y_j\) ensure that no bin is overloaded and no item is assigned to a closed bin.

---

# Problem Challenges motivating the hybrid approach

- The article identifies 1D-BPP as strongly NP-hard, so the repository emphasizes scalable heuristics and metaheuristics rather than exact solution for the evaluated slices.
- Constructive heuristics such as FFD and BFD are fast but deterministic, so they do not use information generated after the initial packing.
- Classical metaheuristics depend on manually designed neighborhoods and fixed operator policies, which motivates learning where the search repeatedly makes choices.
- The hybrid design therefore targets the two repeated ALNS decisions that appear in the implementation: which items to remove and which feasible bin to choose during reinsertion.

---

# Literature Review: ML inside metaheuristics

- The article frames this project within ML-for-combinatorial-optimization, specifically ML used to guide or replace decisions inside a metaheuristic rather than metaheuristics used to tune ML models.
- Adaptive Large Neighborhood Search follows the LNS and ALNS line of work, where destroy and repair operators define very large neighborhoods and adaptive control selects operators over time.
- Contextual bandits such as LinUCB are relevant because they provide lightweight online learning from search-state features without requiring a full offline RL training phase.
- Learning-guided repair is represented in this repository as behavioral cloning, where a supervised model imitates a BFD oracle for local placement decisions.

---

# Literature Review: representative references used by the article

- The article cites Shaw for Large Neighborhood Search and Ropke--Pisinger for Adaptive Large Neighborhood Search as the metaheuristic backbone.
- The article cites Li and Chu for contextual bandit and LinUCB foundations, matching the solver's disjoint LinUCB implementation.
- The article cites Friedman for gradient boosting, matching the repair model implemented with `GradientBoostingClassifier`.
- The article cites Bengio, Lodi, and Prouvost plus Hottung and Tierney to position this work within ML-guided combinatorial search and neural large-neighborhood search.

---

# Proposed Solution — Step 1: Global Architecture

- The global architecture is a single ALNS loop with two explicit ML injection points: online destroy selection before destruction and offline learned bin ranking during repair.
- The data flow begins with item sizes and capacity, constructs a BFD warm start, iteratively proposes candidate packings, and finally converts the best internal solution into `BinPackingSolution`.
- The solver keeps mutable arrays for bins, bin loads, and item-to-bin mappings so destroy and repair operations can update the current solution efficiently.
- The article's Figure 1 and the solver's control flow agree on the same sequence: warm start, context, bandit selection, destroy, repair, acceptance, best-solution tracking, and looping.

<!-- diagram: global architecture -->

```text
Instance (sizes, capacity)
        |
        v
BFD warm start --> current solution + best solution
        |
        v
Build 5-D context ----> Online WarmStartLinUCB ----> destroy arm {random,worst,related}
        |                                               |
        v                                               v
Candidate copy <------------------------------- Destroy k items
        |
        v
Offline GBT repair ranks feasible bins for each displaced item
        |
        v
Consolidate --> SA accept/reject --> update best --> cool/reheat/restart --> next iteration
```

---

# Proposed Solution — Classical ALNS backbone

- The solver initializes `start`, `best`, and `current` from a deterministic BFD-like warm start before entering the iteration loop.
- At every iteration, the solver copies the current solution, chooses an adaptive destruction radius \(k\), destroys the candidate with one of three operators, repairs displaced items, and optionally consolidates bins.
- Candidate acceptance uses simulated annealing, accepting all non-worsening candidates and accepting worsening candidates with probability \(\exp(-\Delta/T)\).
- Termination is controlled by `max_iterations` or an optional wall-clock deadline, and the final stored solution is the best solution rather than merely the last accepted candidate.

---

# Proposed Solution — Destroy and repair mechanics

- The destroy portfolio contains random item removal, worst-load removal from least-filled bins, and related-item removal based on proximity of item sizes.
- The destruction radius is adaptive because stagnation increases the maximum sampled \(k\) from the lower fraction toward the upper fraction of the item count.
- If the offline model is enabled, each displaced item is placed by learned repair; if it is disabled, the ablation fallback is deterministic best-fit repair.
- A post-repair consolidation pass tries to move items out of under-loaded bins into other feasible bins before empty bins are pruned.

---

# Proposed Solution — Acceptance, reheating, and restart

- The default temperature is \(T_0=1/\ln 2\), and the default cooling rate is \(\alpha_{cool}=0.9995\), both exposed as solver parameters.
- The acceptance rule computes \(\Delta=\text{candidate bins}-\text{current bins}\), accepts when \(\Delta\le0\), and otherwise samples according to the simulated-annealing probability.
- Soft reheating raises the temperature during prolonged stagnation, while hard diversification restarts from the best solution after the no-improvement limit is reached.
- The implementation shrinks the patience window after restarts, making later diversification cycles shorter while preserving the global iteration cap.

---

# Proposed Solution — Step 2A: Offline Repair Model role

- The offline repair model solves the ALNS reconstruction subproblem: for each displaced item, it ranks existing feasible bins and selects the bin with the highest predicted probability.
- The learned model was chosen to replace a hand-coded repair choice with a supervised ranking policy that can reuse patterns from offline traces and mid-search states.
- The fallback when the model is disabled is best-fit repair, which confirms that the learned repair is an operator replacement rather than only a post-processing score.
- During inference, a new bin is opened only when no existing bin has enough residual capacity for the item.

---

# Proposed Solution — Offline Repair dataset generation

- The synthetic generator samples random instances with item counts between configured bounds and item sizes drawn from uniform, bimodal, or clipped normal distributions.
- For each generated instance, items are processed in descending-size order and feasible placement candidates are labeled using a BFD oracle.
- The positive label is the feasible bin with minimum post-placement slack, and sampled negative labels are other feasible bins for the same item.
- A separate ALNS-state collection script performs a DAgger-lite augmentation pass by running a model through destroyed states, labeling those on-policy repair states again with the BFD oracle, and saving them as `alns_states`.

---

# Proposed Solution — Offline Repair features and training

- The feature contract is versioned as feature version 2 with 11 features, and the solver rejects model bundles whose feature version or feature count does not match.
- The 11 features describe item size, squared normalized size, size rank, remaining reinsertion progress, bin load, residual capacity, slack after placement, bin occupancy, largest bin item, smallest bin item, and fill ratio.
- The training pipeline loads one or more pickle datasets, performs integrity checks, splits 15 percent as a stratified test set, and trains a `StandardScaler` plus `GradientBoostingClassifier` pipeline.
- The default model hyperparameters include 500 estimators, max depth 6, learning rate 0.03, subsampling 0.75, balanced sample weights, and early stopping through a validation fraction.

---

# Proposed Solution — Offline Repair evaluation and inference integration

- Offline validation uses ROC-AUC, average precision, F1, precision, recall, confusion counts, cross-validation ROC-AUC, and quality gates for minimum ROC-AUC and average precision.
- The saved model bundle contains the classifier, scaler, feature version, feature count, metrics, cross-validation scores, seed, dataset summary, and quality-gate metadata.
- At runtime, the solver builds feature rows for all feasible bins for the current displaced item, applies the stored scaler and classifier, and chooses the feasible bin with maximum predicted class-1 probability.
- The `FastPredictor` optimization manually applies the `StandardScaler` and reads gradient-boosting raw predictions to reduce per-call validation overhead when the expected scikit-learn types are present.

---

# Proposed Solution — Step 2B: Online RL Destroy Operator Selector role

- The online component replaces uniform random destroy-operator selection with a warm-start contextual bandit when `use_online_rl=True`.
- Its action space has three arms corresponding exactly to the implemented destroy operators: random item removal, worst-load removal, and related-item removal.
- Its state is a five-dimensional normalized context containing temperature phase, stagnation progress, cost-to-lower-bound ratio, destruction radius fraction, and iteration progress.
- The solver queries the bandit immediately before dispatching to the selected destroy operator and updates the bandit after candidate acceptance and best-solution checks.

---

# Proposed Solution — Online RL algorithm and reward

- For the first configured warm-up calls, the selector uses Beta--Bernoulli Thompson sampling with per-arm alpha and beta parameters to reduce cold-start risk.
- After warm-up, the selector uses disjoint LinUCB, scoring each arm by \(\hat{\theta}_k^\top x + \alpha\sqrt{x^\top A_k^{-1}x}\).
- The implemented LinUCB update uses a Sherman--Morrison rank-one inverse update and increments the reward vector by \(r x\).
- The reward is 1-scaled improvement when bins are saved, 0.2 for accepted moves without bin savings, and 0 for rejected moves, with improvement normalized by the current gap to \(\mathrm{LB}_1\).

---

# Tests and Results — Protocol and setup

- The article plan reports a fixed random seed of 42, a 300-iteration ALNS budget, \(T_0=1/\ln 2\), \(\alpha_{cool}=0.9995\), and a single pre-trained repair model across datasets.
- The solution-quality metric is the gap \(\text{bins used}-\mathrm{LB}_1\), so a zero gap certifies optimality only relative to the continuous lower bound used in the repository.
- The implementation records runtime as part of benchmark outputs, while the paper reports elapsed solve time for each experimental scenario.
- The article states that experiments were run with Python 3.13, scikit-learn, Windows 11, an Intel Core i9-13950HX CPU, and 64 GB of RAM.

---

# Tests and Results — Datasets

- The test suite in the article covers Scholl-2, Falkenauer-T, Falkenauer-U, Wäscher, and Hard28 benchmark families, and corresponding dataset directories are present under `bin_packing_optimization/datasets`.
- Scholl-2 uses capacity 1000 and 50--500 items with a uniform structure, and the ablation uses the first five 50-item instances.
- Falkenauer-T uses triplet instances at capacity 1000, while Falkenauer-U uses uniform instances with capacity 150.
- Wäscher is described as cutting-stock-style with capacity 10000, and Hard28 is described as hard 160--200-item instances with capacity 1000.

---

# Tests and Results — Parameter calibration

- The article plan places parameter calibration before the ablation, and the repository includes an Optuna-based tuner for ALNS hyperparameters.
- The tuned or fixed online-learning settings reported in the article are LinUCB exploration coefficient \(\alpha=0.3\) and a 300-call Thompson-sampling warm-up.
- The tuner evaluates candidate parameters by running solver configurations on a tuning bank and computing mean gap, mean time, and per-dataset gap summaries.
- This calibration stage answers how the implementation chooses search-control parameters before comparing ML-component variants.

---

# Tests and Results — Ablation study

- The ablation is designed to isolate the two learning components by toggling online RL and offline repair while holding the ALNS settings fixed.
- The four configurations are no learning, online only, offline only, and both combined on five 50-item Scholl-2 instances.
- The reported results are: no learning at 18.2 average bins, 0.20 average gap, 92.76 percent fill, and 0.153 seconds; online only at 18.0 average bins, 0.00 gap, 93.76 percent fill, and 0.186 seconds.
- The reported offline-only and combined configurations both remain at 18.2 average bins and 0.20 average gap, with runtimes of 0.353 and 0.362 seconds, respectively.

---

# Tests and Results — Multi-dataset combined method

- The multi-dataset scenario measures whether the combined dual-learning method remains near the lower bound beyond the small Scholl-2 ablation slice.
- The reported mean gaps are 0.20 for Scholl-2, 1.00 for Falkenauer-T, 0.20 for Falkenauer-U, 0.60 for Wäscher, and 0.67 for Hard28.
- The reported runtime ranges are 0.6--1.2 seconds on Scholl-2, 1.0--1.7 seconds on Falkenauer-T, 4.3--5.8 seconds on Falkenauer-U, 1.0--2.2 seconds on Wäscher, and 13.5--15.1 seconds on Hard28.
- The article interprets Falkenauer-T as difficult relative to \(\mathrm{LB}_1\), because triplet structure can make the continuous lower bound weaker than the true optimum.

---

# Tests and Results — Comparative study

- The comparative study is reported on the same Scholl-2 50-item, five-instance slice as a mean-gap and mean-runtime comparison against implemented team baselines.
- FFD/BFD, simulated annealing, and tabu search each report a mean gap of 1.80, with FFD/BFD under 0.01 seconds, simulated annealing at 0.21 seconds, and tabu search at 0.13 seconds.
- The genetic algorithm reports a mean gap of 1.00 in 0.82 seconds, while ant colony optimization reports a zero mean gap in 3.03 seconds.
- The dual-learning ALNS reports a mean gap of 0.20 in 0.36 seconds, positioning it between the fastest constructive baselines and the slower ant-colony result on this slice.

---

# Synthesis of Results

- The ablation evidence shows that online destroy-operator learning is the decisive contributor on the Scholl-2 slice, because online-only closes the mean gap while adding little runtime.
- The offline repair model is validated as an implemented and trainable component, but the reported ablation shows no quality gain and a clear runtime overhead on the easy five-instance Scholl-2 slice.
- Across five benchmark families, the combined method usually stays within one bin of \(\mathrm{LB}_1\), but this statement is bounded by the small reported sample sizes.
- The collective evidence supports the hybrid ALNS architecture while also showing that the two ML components do not contribute equally under the reported experimental conditions.

---

# Conclusion and future work

- The main contribution is an executable dual-learning ALNS for 1D-BPP that combines classical destroy--repair search, simulated-annealing acceptance, offline learned repair, and online contextual-bandit destroy selection.
- A second contribution is the supervised repair-model pipeline, including synthetic data generation, ALNS-state augmentation, feature-version checks, model training, metrics, and saved model bundles.
- The concrete limitations are small evaluation slices, single-run stochastic baseline reporting, reliance on \(\mathrm{LB}_1\) rather than stronger lower bounds, and offline repair runtime overhead.
- Actionable future work is to evaluate full benchmark families, use stronger bounds such as Martello--Toth \(L_2\), learn repair directly from search reward, and test transfer of the learned components across instance families.
