# Article Spec — Dual-Learning Hybrid ALNS for 1D Bin Packing

Blueprint for the IEEE conference paper. Source of truth: the team's
`approach_explanation.md` (method), `hybrid_alns_performance_evaluation.ipynb`
(results), and the cloned solver code.

## Meta
- **Language:** English
- **Template:** IEEE conference (IEEEtran, two-column), `thebibliography`/BibTeX
- **Target length:** 6 pages
- **Contribution framing:** A — *Dual-learning hybrid ALNS* (two learning
  mechanisms at two decision points; apport isolated by ablation)
- **Authors (Team 4 "SPOT", class SIQ 1):** Mohamed El Amine Kherroubi,
  Rayan Boukakiou, Idriss Yacine Ziadi, Idris Himeur, Adem Abdelhafidh Diar
- **Affiliation:** TBD (likely ESI — confirm with user)

## Working title
"Dual-Learning Adaptive Large Neighborhood Search for One-Dimensional Bin
Packing: Online Operator Selection and Offline Learned Repair"

## Key message
Most ML+metaheuristic hybrids inject learning at a single decision point. We
integrate learning at **two** points of a single ALNS for 1D-BPP — an **online**
contextual bandit (warm-start LinUCB) selecting the destroy operator, and an
**offline** GBT model guiding repair — and quantify each contribution by ablation.

## Section map (IEEE)

### Abstract (~180 words)
Problem (1D-BPP, strongly NP-hard) → limit of metaheuristics (manual operator
design, no learning from search history) → approach (dual-learning hybrid ALNS)
→ ablation + 5-dataset evaluation → key result (online RL alone closes the mean gap
0.20→0.00 on the Scholl-2 slice; offline repair adds cost without quality gain
here). Keywords: bin packing, ALNS, contextual bandit, LinUCB,
gradient boosting, hyper-heuristics, machine learning.

### I. Introduction (6 paragraphs)
1. Problem definition (1D-BPP).
2. SOTA metaheuristics + limits: FFD/BFD (Johnson 11/9), SA (Kirkpatrick), Tabu
   (Glover), GA (Falkenauer), ACO — incl. the team's own implementations. Limits:
   manual operator design, parameter sensitivity, no adaptation from search state.
3. Motivation for ML hybridization + **taxonomy of hybridization modes**
   (ML predicts parameters / ML guides search / ML as operator). Justify chosen mode.
4. SOTA hybridization (ML + metaheuristics for BPP & combinatorial opt:
   RL operator selection, NLNS, learning-to-search; Bengio et al. 2021 survey).
5. Contribution & originality (framing A) + ablation preview.
6. Paper structure.

### II. Mathematical formulation (~0.5 p)
- Items I={1..n}, sizes s_i ∈ Z>0, identical bins capacity C, s_i ≤ C.
- Feasible = partition into bins with Σ_{i∈Bj} s_i ≤ C. Objective min m.
- ILP form: x_ij ∈{0,1} (item i in bin j), y_j ∈{0,1} (bin j used); min Σ y_j s.t.
  Σ_j x_ij = 1 ∀i; Σ_i s_i x_ij ≤ C y_j ∀j; x_ij ≤ y_j.
- LB1 = ⌈Σ s_i / C⌉ (with validity argument). Mention Martello–Toth L2 as sharper.
- Complexity: strongly NP-hard (3-Partition reduction).

### III. Proposed hybrid approach (~1.5 p)
- **Fig. 1** architecture: ALNS loop with the two ML injection points.
- Encoding: bin assignment; working structures (item_to_bin, bin_loads); BFD warm
  start (`_build_ffd_start_solution`, actually BFD).
- LNS/ALNS framework: Shaw 1998 (LNS), Ropke & Pisinger 2006 (ALNS); 3 destroy
  operators (random / worst-load / related-item); adaptive k ∈ [⌊0.05n⌋, ⌊0.25n⌋].
- SA acceptance + geometric cooling (α=0.9995, T0=1/ln2) + soft reheat + restart.
- **ML#1 online — WarmStartLinUCB:** Beta-Bernoulli TS warm-up (300 calls) →
  disjoint LinUCB (Chu et al. 2011, α=0.3); 5-D context [T/T0, stagnation, f/LB,
  k/n, iter/max]; reward = min(1, bins_saved/gap) / 0.2 / 0.
- **ML#2 offline — GBT repair (behavioral cloning from BFD):** 11 features per
  (item,bin); GradientBoostingClassifier + StandardScaler; label=1 if BFD picks
  the bin; trained on fresh BFD traces + post-destruction repair traces.
- **Algorithm 1** pseudocode.

### IV. Experiments, results, analysis (~1.5 p)
- Protocol: Python 3.13, scikit-learn GBT, seed 42; max_iter=300, T0=1/ln2,
  α_cool=0.9995; repair_model_v2.pkl.
- **Table I — datasets:** Scholl-2 (cap 1000), Falkenauer-T (triplets, 1000),
  Falkenauer-U (cap 150), Wäscher (cap 10000), Hard28 (cap 1000).
- Parameter calibration: Optuna pipeline; LinUCB α=0.3, warmup=300.
- **Table II + Fig. 2 — ablation (Scholl-2, 50 items, 5 instances):**

  Aligned run on i9-13950HX (current solver + no-learning bug fix):

  | Method        | avg_bins | avg_gap | fill% | time(s) |
  | ------------- | -------- | ------- | ----- | ------- |
  | no-learning   | 18.2     | 0.20    | 92.76 | 0.153   |
  | online-only   | 18.0     | 0.00    | 93.76 | 0.186   |
  | offline-only  | 18.2     | 0.20    | 92.76 | 0.353   |
  | both-combined | 18.2     | 0.20    | 92.76 | 0.362   |

  Findings: online RL alone reaches gap 0 (best config); offline repair adds
  runtime without quality gain on this slice (slightly degrades online-only).
- **Table III — multi-dataset (combined), gap-to-LB + time:**
  - Scholl-2 (5×50): gaps {0,1,0,0,0}, avg 0.2; ~0.6–1.2 s
  - Falkenauer-T (5×60): gaps all 1 (21 vs LB 20); ~1.0–1.7 s
  - Falkenauer-U (5×120, cap 150): gaps {0,0,0,1,0}, avg 0.2; ~4.3–5.8 s
  - Wäscher (5, cap 10000): gaps {0,0,1,1,1}, avg 0.6; ~1.0–2.2 s
  - Hard28 (3×160): gaps {0,1,1}, avg 0.667; ~13.5–15.1 s
- **Table IV — comparative study vs team methods** (Scholl-2, 50 items, 5 inst.,
  same aligned i9 session via `make_tables.py`): FFD/BFD 1.80/<0.01, SA 1.80/0.21,
  Tabu 1.80/0.13, GA 1.00/0.82, ACO 0.00/3.03, Dual-learning ALNS 0.20/0.36.
  Exact methods omitted (intractable at n=50). Metaheuristic baselines are
  stochastic (single run reported).

### V. Conclusion & perspectives (~0.3 p)
Dual learning helps; online operator selection is the cheap win, learned repair
trades runtime for placement quality. Limits: small evaluation slices (5 instances),
offline model runtime. Future: scaling to full datasets, deeper RL, transfer across
instance families.

### VI. References (BibTeX, ~15–20)
BPP surveys (Coffman et al.; Delorme, Iori, Martello 2016), FFD/BFD (Johnson 1973),
SA (Kirkpatrick 1983), Tabu (Glover 1986), GA for BPP (Falkenauer 1996), ACO,
LNS (Shaw 1998), ALNS (Ropke & Pisinger 2006), LinUCB (Li et al. 2010; Chu et al.
2011), gradient boosting (Friedman 2001), ML+CO survey (Bengio, Lodi, Prouvost 2021),
NLNS (Hottung & Tierney 2020), datasets (Scholl et al.; Falkenauer; Wäscher; Schwerin
& Wäscher / Hard28).

## Figures / Tables
- Fig. 1 — Hybrid ALNS architecture (two ML injection points) — TikZ.
- Fig. 2 — Ablation: avg gap & fill by method — from notebook graphs / regenerate.
- Table I — Datasets. Table II — Ablation. Table III — Multi-dataset. Table IV — vs methods.

## Deliverable layout
`OPTIM/article/` : `main.tex`, `refs.bib`, `IEEEtran.cls` (copied), `figures/`.
Standalone LaTeX project (not committed to the third-party repo).

## Open items
- Confirm affiliation/institution for the IEEE author block.
- Decide Table IV data source (extract from team notebooks vs re-run).
