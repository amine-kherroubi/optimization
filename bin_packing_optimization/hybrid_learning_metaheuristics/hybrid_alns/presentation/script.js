const slides = [
  {
    kind: "content",
    title: "Hybrid ALNS for the Bin Packing Problem",
    content:
      "<h1>Hybrid ALNS for the Bin Packing Problem</h1>\n<h3>One-Dimensional Variant</h3>\n<p>A solver combining metaheuristic large-neighborhood search, adaptive bandit-based operator selection, and machine-learned repair via behavioral cloning from BFD.</p>",
  },
  {
    kind: "content",
    title: "Introduction",
    content:
      "<h2>Introduction</h2>\n<p><strong>Project context:</strong> The one-dimensional Bin-Packing Problem (1D-BPP) arises across many industrial domains — cutting stock, vehicle loading, container logistics, and cloud resource allocation — wherever discrete items must be packed into fixed-capacity bins at minimum cost.</p>\n<p><strong>Why classical methods fall short:</strong> 1D-BPP is strongly NP-hard. Exact solvers (branch-and-price) are optimal but scale to only a few hundred items. Classical metaheuristics scale well but rely on fixed decision rules that do not adapt to the problem structure encountered at runtime, leading to stagnation in dense local optima.</p>\n<p><strong>Motivation for the hybrid approach:</strong> Machine learning offers a principled way to make metaheuristic decisions adaptive — without replacing the optimization loop:</p>\n<ul><li><strong>Online</strong> — learn <em>which</em> destroy operator performs best given the current search context (contextual bandit)</li><li><strong>Offline</strong> — learn <em>which</em> bin to assign each displaced item by imitating an expert heuristic (behavioral cloning)</li></ul>\n<p><strong>This project</strong> embeds both components inside an Adaptive Large-Neighborhood Search (ALNS) framework, keeping them independently togglable to enable clean ablation. The goal is to demonstrate that ML-guided decisions can meaningfully improve solution quality and adaptivity over non-learning ALNS.</p>",
  },
  {
    kind: "content",
    title: "Outline",
    content:
      "<h2>Outline</h2>\n<ol><li><strong>Introduction</strong> — project context and hybrid motivation</li><li><strong>Problem Definition</strong> — formal model, lower bound, NP-hardness</li><li><strong>Literature Review</strong> — ML inside metaheuristics</li><li><strong>Proposed Solution</strong><ul><li>4a. Global architecture overview</li><li>4b. BFD warm start and ALNS framework</li><li>4c. ML Component I — LinUCB bandit for operator selection</li><li>4d. ML Component II — GBT learned repair</li></ul></li><li><strong>Tests &amp; Results</strong> — protocol, benchmarks, ablation, multi-dataset, comparative</li><li><strong>Synthesis &amp; Conclusion</strong> — findings, contributions, limitations, future work</li></ol>",
  },
  {
    kind: "content",
    title: "I.1 Problem Definition",
    content:
      '<h2>I.1 Problem Definition</h2>\n<p><strong>Given:</strong> item set $\\mathcal{I} = \\{1, \\ldots, n\\}$ with integer sizes $s_i \\in \\mathbb{Z}_{>0}$; identical bins of integer capacity $C \\in \\mathbb{Z}_{>0}$, with $s_i \\le C$ for all $i$.</p>\n<p><strong>Feasible solution</strong> — a partition $B_1, \\ldots, B_m$ of $\\mathcal{I}$ satisfying:</p>\n<div class="formula-card inline-formula"><div class="formula-katex">$$\\biguplus_{j=1}^{m} B_j = \\mathcal{I}, \\qquad \\sum_{i \\in B_j} s_i \\le C \\quad \\forall\\, j \\in \\{1,\\ldots,m\\}$$</div></div>\n<p><strong>Objective:</strong></p>\n<div class="formula-card inline-formula"><div class="formula-katex">$$\\min_{B_1,\\ldots,B_m} \\; m$$</div></div>',
  },
  {
    kind: "content",
    title: "I.2 Continuous Relaxation Lower Bound",
    content:
      '<h2>I.2 Continuous Relaxation Lower Bound</h2>\n<div class="formula-card inline-formula"><div class="formula-katex">$$\\mathrm{LB}_1 = \\left\\lceil \\frac{\\displaystyle\\sum_{i=1}^{n} s_i}{C} \\right\\rceil$$</div></div>\n<p><strong>Proof.</strong> Any feasible packing must hold all item volume. With $m$ bins each contributing at most $C$ units, we need $mC \\ge \\sum_i s_i$, giving $m \\ge \\sum_i s_i / C$. Integer rounding yields the ceiling. $\\square$</p>\n<blockquote><p>Tighter bounds exist (e.g. the Martello–Toth $L_2$ bound, which accounts for large items that cannot coexist). $\\mathrm{LB}_1$ is sufficient as a progress indicator in this implementation.</p></blockquote>',
  },
  {
    kind: "content",
    title: "I.3 Complexity and Algorithmic Strategy",
    content:
      '<h2>I.3 Complexity and Algorithmic Strategy</h2>\n<p>1D-BPP is <strong>strongly NP-hard</strong> by reduction from 3-Partition.</p>\n<div class="table-card"><table><thead><tr><th>Approach</th><th>Representative</th><th>Guarantee</th><th>Scalability</th></tr></thead><tbody><tr><td>Exact</td><td>Branch-and-bound, branch-and-price</td><td>Optimal</td><td>Up to a few hundred items</td></tr><tr><td>Approximation</td><td>FFD, BFD</td><td>$\\le \\frac{11}{9}\\,\\mathrm{OPT} + O(1)$, in $O(n \\log n)$</td><td>High</td></tr><tr><td>Metaheuristic</td><td>ALNS (this solver)</td><td>None (heuristic)</td><td>High</td></tr></tbody></table></div>\n<p><strong>Strategy:</strong> use BFD as a deterministic warm start, then apply ALNS to escape local optima and push toward the lower bound.</p>',
  },
  {
    kind: "content",
    title: "II. Literature Review — ML in Metaheuristics",
    content:
      "<h2>II. Literature Review — ML in Metaheuristics</h2>\n<blockquote><p><strong>Scope:</strong> ML <em>inside</em> metaheuristics — using learned models to guide decisions within the optimization loop. This is distinct from \"metaheuristics for ML\" (hyperparameter tuning, NAS), which is outside this project's scope.</p></blockquote>\n<p><strong>Stream 1 — Adaptive Operator Selection (AOS)</strong></p>\n<p>Bandit-based AOS replaces static operator probabilities with online reward signals. Fialho et al. (2010) formalised this as a multi-armed bandit; subsequent work (COMPASS, Maturana &amp; Saubion 2008) embedded it in evolutionary and population-based frameworks. <strong>This project uses LinUCB (Chu et al. 2011)</strong> — a contextual extension that conditions operator selection on search-state features, enabling fine-grained adaptation.</p>\n<p><strong>Stream 2 — Learned Repair / Construction</strong></p>\n<p>Khalil et al. (2017) demonstrated that graph neural networks can learn greedy construction policies competitive with classical heuristics on TSP, MVC, and MAXCUT. Behavioral cloning (imitation learning) trains a policy directly from expert demonstrations without a reward signal — applied here to replicate BFD's bin-assignment logic under partial solutions.</p>\n<p><strong>Stream 3 — ML-Augmented LNS</strong></p>\n<p>Hottung &amp; Tierney (2020) and Lu et al. (2021) use deep RL to learn full destroy-and-repair operators for CVRP and VRPTW. This project takes a lighter, more interpretable approach: a GBT ranker trained offline, combined with an online bandit for operator selection.</p>\n<p><strong>Positioning of this work:</strong> at the intersection of Streams 1 and 2 — dual-learning ALNS combining online contextual bandit selection with offline supervised repair.</p>",
  },
  {
    kind: "dividerContent",
    title: "III. Proposed Solution",
    content:
      "<h2>III. Proposed Solution</h2>\n<h3>Global Architecture · BFD Warm Start · ALNS Framework · ML Components</h3>",
  },
  {
    kind: "content",
    title: "III.0 Global Architecture",
    content:
      "<h2>III.0 Global Architecture</h2>\n<p><strong>End-to-end pipeline of the hybrid ALNS solver:</strong></p>\n<pre><code>┌─────────────────────────────────────────────────────────────┐\n│                    Problem Instance                          │\n│              (n items, sizes sᵢ, capacity C)                 │\n└────────────────────────┬────────────────────────────────────┘\n                         ▼\n           ┌─────────────────────────┐\n           │    BFD Initialization   │  ← Constructive warm start\n           │  Sort → Best-Fit Decr.  │\n           └────────────┬────────────┘\n                        │  x₀ (initial solution)\n                        ▼\n┌─────────────────────────────────────────────────────────────┐\n│                      ALNS Main Loop                          │\n│                                                              │\n│  ┌────────────────┐  operator  ┌────────────────────────┐   │\n│  │   LinUCB       │◄───────────│    Context Vector       │   │\n│  │   Bandit (ML)  │            │  [T, stagnation, gap,  │   │\n│  └───────┬────────┘            │   k/n, iter progress]  │   │\n│          │ select operator     └────────────────────────┘   │\n│          ▼                                                   │\n│  ┌────────────────┐            ┌────────────────────────┐   │\n│  │  Destroy Step  │─── x̂ ─────►│   GBT Repair (ML)      │   │\n│  │  (Random /     │            │  Score feasible        │   │\n│  │  Worst-load /  │            │  (item, bin) pairs     │   │\n│  │  Related-item) │            └───────────┬────────────┘   │\n│  └────────────────┘                        │ x'              │\n│                                            ▼                 │\n│                          ┌─────────────────────────────┐    │\n│                          │   SA Acceptance Criterion    │    │\n│                          │   Accept x' or keep x        │    │\n│                          │   Update x_best              │    │\n│                          └──────────────┬──────────────┘    │\n│                                         │ reward             │\n│                                         └──────► LinUCB      │\n│                                                   update      │\n└───────────────────────────┬─────────────────────────────────┘\n                            ▼\n                 ┌─────────────────────┐\n                 │   Best Solution x*  │\n                 │  gap = bins − LB₁   │\n                 └─────────────────────┘</code></pre>\n<p><strong>Two ML decision points:</strong> (1) LinUCB selects the destroy operator at each iteration based on search context; (2) GBT scores all feasible bin candidates during repair. Both components are independently togglable for ablation.</p>",
  },
  {
    kind: "content",
    title: "III.1 Constructive Initialization: Best-Fit Decreasing",
    content:
      "<h2>III.1 Constructive Initialization: Best-Fit Decreasing</h2>\n<p>BFD builds a tight deterministic starting point before ALNS begins.</p>\n<ol><li>Sort items by <strong>non-increasing size</strong> (ties broken by item index)</li><li>For each item, evaluate all currently open bins that can accommodate it</li><li>Place the item into the feasible bin with <strong>minimum post-placement slack</strong></li><li>If no feasible bin exists, open a new bin</li></ol>\n<blockquote><p>A tighter warm start leaves fewer bins for ALNS to improve from, which tends to improve final solution quality and reduces time spent recovering from a weak initial packing.</p></blockquote>",
  },
  {
    kind: "dividerContent",
    title: "III.2 Metaheuristic Framework",
    content:
      "<h2>III.2 Metaheuristic Framework</h2>\n<h3>Adaptive Large-Neighborhood Search</h3>",
  },
  {
    kind: "content",
    title: "III.2a Why Local Search Alone Fails",
    content:
      "<h2>III.2a Why Local Search Alone Fails</h2>\n<p>Classical local search applies small moves within a neighborhood $\\mathcal{N}(x)$ — for example, relocating a single item. While efficient, it stalls at <strong>local optima</strong>: solutions from which no single move improves the objective, yet which are far from globally optimal.</p>\n<ul><li>Local optima in 1D-BPP are <strong>dense</strong>, tied to specific item groupings within bins</li><li>No bounded sequence of single-item relocations can reliably escape them</li></ul>\n<p><strong>Solution — Large-Neighborhood Search (Shaw, 1998):</strong> Operate on implicitly exponential neighborhoods by partially destroying and then repairing the current solution.</p>",
  },
  {
    kind: "content",
    title: "III.2b The LNS Iterate",
    content:
      "<h2>III.2b The LNS Iterate</h2>\n<p>Each iteration consists of two phases:</p>\n<p><strong>Destroy</strong> — remove a subset $D$ of items from their assigned bins, producing a partial solution $\\hat{x}$.</p>\n<p><strong>Repair</strong> — reinsert every item in $D$ into $\\hat{x}$, restoring feasibility and yielding a new complete solution $x'$.</p>\n<p>The neighborhood is implicitly exponential in $|D|$: repair can yield any feasible completion of $\\hat{x}$, allowing escape from local optima unreachable by any sequence of small moves.</p>\n<p>ALNS (Ropke &amp; Pisinger, 2006) extends LNS by <strong>adaptively selecting</strong> the destroy operator from a portfolio.</p>",
  },
  {
    kind: "content",
    title: "III.2c Destroy Operators",
    content:
      '<h2>III.2c Destroy Operators</h2>\n<p>Three operators are available. All guarantee that at least one item remains placed.</p>\n<div class="table-card"><table><thead><tr><th>Arm</th><th>Operator</th><th>Mechanism</th></tr></thead><tbody><tr><td>0</td><td><strong>Random</strong></td><td>Sample exactly $\\min(k,\\,n-1)$ items uniformly at random across all placed items and remove them</td></tr><tr><td>1</td><td><strong>Worst-load</strong></td><td>Sort bins by ascending load (small uniform tie-breaking perturbation). Collect items in that order until $\\min(k,\\,n-1)$ items have been removed</td></tr><tr><td>2</td><td><strong>Related-item</strong></td><td>Select a seed item uniformly at random. Remove the seed plus the $\\min(k-1,\\,n-2)$ items with the smallest absolute size difference from the seed</td></tr></tbody></table></div>\n<p>No single operator dominates: random destruction explores broadly early on, while targeted strategies are more effective near a local optimum.</p>',
  },
  {
    kind: "content",
    title: "III.2d Destruction Radius",
    content:
      '<h2>III.2d Destruction Radius</h2>\n<p>The displaced-item count $k$ is drawn uniformly from $[k_{\\min}, k_{\\max}]$ at each iteration:</p>\n<div class="formula-card inline-formula"><div class="formula-katex">$$k_{\\min} = \\max\\!\\left(1,\\,\\left\\lfloor 0.05\\,n \\right\\rfloor\\right), \\qquad k_{\\max} = \\max\\!\\left(k_{\\min}+1,\\,\\left\\lfloor 0.25\\,n \\right\\rfloor\\right)$$</div></div>\n<p>So between <strong>5% and 25%</strong> of items are displaced per iteration.</p>\n<p><strong>Adaptive expansion.</strong> As <code>iterations_since_improvement</code> grows toward <code>no_improve_limit</code>, $k_{\\max}$ interpolates linearly upward, broadening the destruction radius to encourage diversification when the search stagnates.</p>',
  },
  {
    kind: "dividerContent",
    title: "III.3 Acceptance Mechanism",
    content:
      "<h2>III.3 Acceptance Mechanism</h2>\n<h3>Simulated Annealing with Soft Reheat and Restart</h3>",
  },
  {
    kind: "content",
    title: "III.3a SA Acceptance and Cooling",
    content:
      '<h2>III.3a SA Acceptance and Cooling</h2>\n<p>Let $\\Delta = f(x\') - f(x)$. Accept $x\'$ if:</p>\n<div class="formula-card inline-formula"><div class="formula-katex">$$\\Delta \\le 0 \\quad \\text{or} \\quad U < \\exp\\!\\left(-\\Delta/T\\right), \\qquad U \\sim \\mathrm{Uniform}(0,1)$$</div></div>\n<p><strong>Geometric cooling</strong> is applied after every iteration:</p>\n<div class="formula-card inline-formula"><div class="formula-katex">$$T \\leftarrow \\alpha_{\\mathrm{cool}} \\cdot T, \\quad 0 < \\alpha_{\\mathrm{cool}} \\le 1$$</div></div>\n<p><strong>Soft reheat.</strong> To prevent temperature collapse during prolonged stagnation, whenever <code>iterations_since_improvement</code> is a positive multiple of $\\max\\!\\left(50,\\,\\lfloor\\mathrm{no\\_improve\\_limit}/4\\rfloor\\right)$:</p>\n<div class="formula-card inline-formula"><div class="formula-katex">$$T \\leftarrow \\max(T,\\;0.35\\cdot T_0)$$</div></div>',
  },
  {
    kind: "content",
    title: "III.3b Diversification Restart",
    content:
      '<h2>III.3b Diversification Restart</h2>\n<p>When <code>iterations_since_improvement</code> reaches <code>no_improve_limit</code>:</p>\n<ol><li>The current solution is reset to the <strong>best solution found so far</strong></li><li>Temperature is <strong>reheated</strong>: $T \\leftarrow \\max(T,\\;0.20\\cdot T_0)$</li><li>The patience window <strong>shrinks</strong>:</li></ol>\n<div class="formula-card inline-formula"><div class="formula-katex">$$\\mathrm{no\\_improve\\_limit} \\leftarrow \\max\\!\\left(100,\\,\\left\\lfloor\\tfrac{2}{3}\\,\\mathrm{no\\_improve\\_limit}\\right\\rfloor\\right)$$</div></div>\n<p>so successive restarts trigger progressively sooner</p>\n<ol start="4"><li><code>iterations_since_improvement</code> is reset to zero</li></ol>\n<p>The outer budget <code>max_iterations</code> is the <strong>sole hard termination criterion</strong>; the restart mechanism never terminates the run.</p>',
  },
  {
    kind: "dividerContent",
    title: "IV. Machine Learning Components",
    content:
      "<h2>IV. Machine Learning Components</h2>\n<h3>Operator Selection + Learned Repair</h3>",
  },
  {
    kind: "content",
    title: "IV. Two Independent ML Components",
    content:
      '<h2>IV. Two Independent ML Components</h2>\n<div class="table-card"><table><thead><tr><th>Component</th><th>Decision Targeted</th><th>Method</th></tr></thead><tbody><tr><td><strong>Contextual Bandit</strong></td><td>Which destroy operator to apply each iteration</td><td>Warm-start LinUCB</td></tr><tr><td><strong>Repair Ranker</strong></td><td>Which bin to assign each displaced item</td><td>Behavioral cloning (GBT)</td></tr></tbody></table></div>\n<p>The two components are <strong>independent</strong> in implementation and can be enabled or disabled separately. Component I selects the operator used during destruction; Component II guides the subsequent repair.</p>',
  },
  {
    kind: "content",
    title: "IV.1 Operator Selection: Phase 1",
    content:
      "<h2>IV.1 Operator Selection: Phase 1</h2>\n<h3>Warm-up — Beta-Bernoulli Thompson Sampling (first 300 calls)</h3>\n<p>All arms initialised at $\\mathrm{Beta}(1,1)$ (uniform prior).</p>\n<p>For each call:</p>\n<ol><li>Sample $\\tilde{\\theta}_k \\sim \\mathrm{Beta}(\\alpha_k,\\beta_k)$ for each arm $k \\in \\{0,1,2\\}$</li><li>Select $k^* = \\arg\\max_k \\tilde{\\theta}_k$</li><li>Observe reward $r \\in [0,1]$; draw $\\tilde{r} \\sim \\mathrm{Bernoulli}(r)$, then update $k^*$ only:<ul><li>$\\tilde{r} = 1 \\;\\Rightarrow\\; \\alpha_{k^*} \\leftarrow \\alpha_{k^*}+1$</li><li>$\\tilde{r} = 0 \\;\\Rightarrow\\; \\beta_{k^*} \\leftarrow \\beta_{k^*}+1$</li></ul></li></ol>\n<p>This phase identifies well-performing operators without requiring the context vector, mitigating the cold-start problem of pure LinUCB on short runs.</p>",
  },
  {
    kind: "content",
    title: "IV.1 Operator Selection: Phase 2",
    content:
      '<h2>IV.1 Operator Selection: Phase 2</h2>\n<h3>Exploitation — Disjoint LinUCB (calls 301+, $\\alpha = 0.3$, Chu et al. 2011)</h3>\n<p>Each arm $k$ maintains $A_k^{-1} \\in \\mathbb{R}^{5\\times5}$ and $\\mathbf{b}_k \\in \\mathbb{R}^5$.</p>\n<p><strong>Selection.</strong> Given context $\\mathbf{x} \\in \\mathbb{R}^5$:</p>\n<div class="formula-card inline-formula"><div class="formula-katex">$$k^* = \\arg\\max_k \\Bigl[\\hat{\\boldsymbol{\\theta}}_k^\\top\\mathbf{x} + \\alpha\\sqrt{\\mathbf{x}^\\top A_k^{-1}\\mathbf{x}}\\Bigr], \\qquad \\hat{\\boldsymbol{\\theta}}_k = A_k^{-1}\\mathbf{b}_k$$</div></div>\n<p><strong>Update</strong> via Sherman-Morrison rank-1 formula ($O(d^2)$):</p>\n<div class="formula-card inline-formula"><div class="formula-katex">$$A_{k^*}^{-1} \\leftarrow A_{k^*}^{-1} - \\frac{(A_{k^*}^{-1}\\mathbf{x})(A_{k^*}^{-1}\\mathbf{x})^\\top}{1+\\mathbf{x}^\\top A_{k^*}^{-1}\\mathbf{x}}, \\qquad \\mathbf{b}_{k^*} \\leftarrow \\mathbf{b}_{k^*} + r\\,\\mathbf{x}$$</div></div>',
  },
  {
    kind: "content",
    title: "IV.1 Context Vector",
    content:
      '<h2>IV.1 Context Vector</h2>\n<p>The 5-dimensional feature $\\mathbf{x}$ encodes the current search state:</p>\n<div class="table-card"><table><thead><tr><th>Index</th><th>Feature</th><th>Description</th></tr></thead><tbody><tr><td>0</td><td>$T / T_0$</td><td>Normalised temperature; 1 = start (hot), 0 = cold</td></tr><tr><td>1</td><td><code>iter_no_improve</code> / <code>no_improve_limit</code></td><td>Stagnation progress toward the next restart</td></tr><tr><td>2</td><td>$\\mathrm{LB}_1 / f(x)$</td><td>Lower-bound-to-cost ratio; 1.0 = optimal</td></tr><tr><td>3</td><td>$k / n$</td><td>Current destruction radius as a fraction of $n$</td></tr><tr><td>4</td><td><code>iteration</code> / <code>max_iterations</code></td><td>Overall search progress (0 to 1)</td></tr></tbody></table></div>\n<p>All features lie in $[0,1]$, enabling the UCB exploration term to be comparable across dimensions without additional normalisation.</p>',
  },
  {
    kind: "content",
    title: "IV.1 Reward Signal",
    content:
      '<h2>IV.1 Reward Signal</h2>\n<p>Let $\\mathrm{gap} = \\max(1,\\,f(x) - \\mathrm{LB}_1)$ be the current gap after updating the incumbent.</p>\n<div class="formula-card inline-formula"><div class="formula-katex">$$r = \\begin{cases} \\min\\!\\left(1,\\;\\dfrac{\\max(0,-\\Delta)}{\\mathrm{gap}}\\right) & \\text{bins were saved } (\\Delta < 0) \\\\[8pt] 0.2 & \\text{accepted without saving bins} \\\\[6pt] 0.0 & \\text{rejected} \\end{cases}$$</div></div>\n<blockquote><p>Normalising by $\\mathrm{gap}$ rewards bin savings more highly when the solution is already close to the lower bound, reflecting the increasing marginal difficulty of further improvement.</p></blockquote>',
  },
  {
    kind: "content",
    title: "IV.2 Machine-Learned Repair: Overview",
    content:
      "<h2>IV.2 Machine-Learned Repair: Overview</h2>\n<p>After destruction, each displaced item $i$ must be reinserted. Let $\\mathcal{F}(i) = \\{j : \\ell_j + s_i \\le C\\}$ be the set of feasible bins.</p>\n<ul><li>$\\mathcal{F}(i) = \\emptyset$: open a new bin immediately</li><li>Otherwise: <strong>score every bin</strong> in $\\mathcal{F}(i)$; place item $i$ in the <strong>highest-scoring bin</strong></li><li>Items processed in <strong>non-increasing size order</strong></li></ul>\n<p><strong>Training objective:</strong> binary classifier on feasible $(\\text{item},\\text{bin})$ pairs. Label 1 if BFD would have chosen that bin, label 0 otherwise. This is behavioral cloning from the BFD expert.</p>",
  },
  {
    kind: "content",
    title: "IV.2 Feature Representation",
    content:
      '<h2>IV.2 Feature Representation</h2>\n<p>Each feasible $(i,j)$ pair is encoded as an <strong>11-dimensional vector</strong> (all features normalised by $C$ or $n$):</p>\n<div class="table-card"><table><thead><tr><th>Idx</th><th>Feature</th><th>Description</th></tr></thead><tbody><tr><td>0</td><td>$s_i / C$</td><td>Normalised item size</td></tr><tr><td>1</td><td>$(s_i / C)^2$</td><td>Squared normalised size</td></tr><tr><td>2</td><td>$\\mathrm{rank}(i) / n$</td><td>Size rank among $n$ items (0 = largest)</td></tr><tr><td>3</td><td>$\\mathrm{remaining} / n$</td><td>Fraction of displaced items not yet reinserted</td></tr><tr><td>4</td><td>$\\ell_j / C$</td><td>Normalised current bin load</td></tr><tr><td>5</td><td>$(C - \\ell_j) / C$</td><td>Residual capacity fraction</td></tr><tr><td>6</td><td>$(C - \\ell_j - s_i) / C$</td><td>Post-placement slack fraction</td></tr><tr><td>7</td><td>$\\lvert B_j \\rvert / n$</td><td>Normalised bin occupancy (item count)</td></tr><tr><td>8</td><td>$\\max_{k \\in B_j} s_k / C$</td><td>Largest item already in bin $j$</td></tr><tr><td>9</td><td>$\\min_{k \\in B_j} s_k / C$</td><td>Smallest item already in bin $j$</td></tr><tr><td>10</td><td>$s_i / (C - \\ell_j)$</td><td>Fill ratio: fraction of residual capacity consumed by item $i$</td></tr></tbody></table></div>',
  },
  {
    kind: "content",
    title: "IV.2 Model Architecture and Training Data",
    content:
      "<h2>IV.2 Model Architecture and Training Data</h2>\n<p><strong>Repair ranker:</strong> <code>GradientBoostingClassifier</code> (scikit-learn)</p>\n<ul><li><code>StandardScaler</code> fit on training features, serialised with the model</li><li>Inference via <code>_FastPredictor</code>: scaler applied in NumPy, <code>_raw_predict</code> called directly on the GBT — bypasses per-call validation; logits converted to probabilities via sigmoid</li><li>Bundle stores <code>feature_version</code> and <code>n_features</code>; mismatch raises an error at load time</li><li>Balanced sample weights; primary metric: <strong>ROC-AUC</strong> (probabilities used for bin ranking, not hard class labels)</li></ul>\n<p><strong>Training data</strong> — two trace types pooled to reduce distribution shift:</p>\n<ul><li><strong>Fresh BFD traces:</strong> replay BFD from scratch; label chosen bin positive, sample remaining feasible bins negative</li><li><strong>Post-destruction traces:</strong> build a BFD solution, evict a random fraction of bins, reinsert displaced items with the same labeling rule</li></ul>",
  },
  {
    kind: "dividerContent",
    title: "V. Experimental Evaluation",
    content:
      "<h2>V. Experimental Evaluation</h2>\n<h3>Protocol · Benchmarks · Ablation · Results</h3>",
  },
  {
    kind: "content",
    title: "V.1 Experimental Protocol",
    content:
      '<h2>V.1 Experimental Protocol</h2>\n<div class="table-card"><table><thead><tr><th>Setting</th><th>Value</th></tr></thead><tbody><tr><td>Random seed</td><td>42</td></tr><tr><td>ALNS iterations</td><td>300</td></tr><tr><td>Initial temperature $T_0$</td><td>$1/\\ln 2$</td></tr><tr><td>Cooling coefficient $\\alpha_{\\mathrm{cool}}$</td><td>0.9995</td></tr><tr><td>LinUCB exploration $\\alpha$</td><td>0.3</td></tr><tr><td>Thompson-sampling warm-up</td><td>300 calls</td></tr><tr><td>Repair model</td><td>Single pre-trained bundle (all datasets)</td></tr></tbody></table></div>\n<p><strong>Primary metric throughout:</strong></p>\n<div class="formula-card inline-formula"><div class="formula-katex">$$\\text{gap} = \\text{bins used} - \\mathrm{LB}_1 \\qquad (\\text{gap} = 0 \\;\\Leftrightarrow\\; \\text{LB-certifiably optimal})$$</div></div>\n<p><strong>Environment:</strong> Python 3.13 · scikit-learn · Windows 11 · Intel i9-13950HX · 64 GB RAM</p>',
  },
  {
    kind: "content",
    title: "V.2 Benchmark Datasets — Instance Characteristics",
    content:
      '<h2>V.2 Benchmark Datasets — Instance Characteristics</h2>\n<p>Five families covering structurally distinct regimes:</p>\n<div class="table-card"><table><thead><tr><th>Family</th><th>Capacity $C$</th><th>Items $n$</th><th>Structure</th></tr></thead><tbody><tr><td>Scholl-2</td><td>1 000</td><td>50 – 500</td><td>Uniform sizes</td></tr><tr><td>Falkenauer-T</td><td>1 000</td><td>—</td><td>Triplet structure</td></tr><tr><td>Falkenauer-U</td><td>150</td><td>—</td><td>Uniform sizes</td></tr><tr><td>Wäscher</td><td>10 000</td><td>—</td><td>Cutting-stock</td></tr><tr><td>Hard28</td><td>1 000</td><td>160 – 200</td><td>Adversarially hard</td></tr></tbody></table></div>\n<blockquote><p><strong>Falkenauer-T note:</strong> the triplet structure systematically weakens $\\mathrm{LB}_1$ relative to the true optimum. Reported gaps for this family are <strong>not directly comparable</strong> to those of other families.</p></blockquote>',
  },
  {
    kind: "content",
    title: "V.3 Test 1 — Ablation Study",
    content:
      '<h2>V.3 Test 1 — Ablation Study</h2>\n<p><strong>Purpose of this test:</strong> Isolate the individual contribution of each ML component (LinUCB bandit and GBT repair ranker) by toggling each independently. This determines which component drives quality improvement and which adds computational overhead without proportional benefit, validating the design choices of the hybrid architecture.</p>\n<p><strong>Scholl-2 · 5 instances · 50 items</strong> — each component toggled independently.</p>\n<div class="table-card"><table><thead><tr><th>Configuration</th><th>Avg. bins</th><th>Avg. gap</th><th>Fill %</th><th>Time (s)</th></tr></thead><tbody><tr><td>No learning (baseline)</td><td>18.2</td><td>0.20</td><td>92.76</td><td>0.153</td></tr><tr><td><strong>Online RL only</strong></td><td><strong>18.0</strong></td><td><strong>0.00</strong></td><td><strong>93.76</strong></td><td><strong>0.186</strong></td></tr><tr><td>Offline GBT only</td><td>18.2</td><td>0.20</td><td>92.76</td><td>0.353</td></tr><tr><td>Both combined</td><td>18.2</td><td>0.20</td><td>92.76</td><td>0.362</td></tr></tbody></table></div>\n<p><strong>Key finding:</strong> the online RL selector is the <strong>decisive contributor</strong> — it closes the gap to 0 alone. The offline GBT repair model adds approximately <strong>2× runtime</strong> with no quality gain on this small-instance slice; its benefit is expected to emerge at larger scale.</p>\n<blockquote><p>Conclusions are bounded by a 5-instance evaluation slice at a single instance size. Large-instance behaviour is an open empirical question.</p></blockquote>',
  },
  {
    kind: "content",
    title: "V.4 Test 2 — Multi-Dataset Generalization",
    content:
      '<h2>V.4 Test 2 — Multi-Dataset Generalization</h2>\n<p><strong>Purpose of this test:</strong> Evaluate the generalization of the combined method (online RL + offline GBT) across structurally distinct benchmark families. Assesses whether the hybrid approach adapts to different item size distributions, capacity regimes, and structural properties — including adversarially hard instances — without retuning any parameter.</p>\n<p><strong>Combined method</strong> (online RL + offline GBT repair) across all benchmark families.</p>\n<div class="table-card"><table><thead><tr><th>Family</th><th>Mean gap</th><th>Runtime (s)</th></tr></thead><tbody><tr><td>Scholl-2</td><td>0.20</td><td>0.6 – 1.2</td></tr><tr><td>Falkenauer-T</td><td>1.00</td><td>1.0 – 1.7</td></tr><tr><td>Falkenauer-U</td><td>0.20</td><td>4.3 – 5.8</td></tr><tr><td>Wäscher</td><td>0.60</td><td>1.0 – 2.2</td></tr><tr><td>Hard28</td><td>0.67</td><td>13.5 – 15.1</td></tr></tbody></table></div>\n<p><strong>4 of 5 families</strong> land within 1 bin of $\\mathrm{LB}_1$. The Falkenauer-T gap of 1.00 most likely reflects the weakness of $\\mathrm{LB}_1$ on triplet instances rather than degraded search quality.</p>',
  },
  {
    kind: "content",
    title: "V.5 Test 3 — Comparative Study",
    content:
      '<h2>V.5 Test 3 — Comparative Study</h2>\n<p><strong>Purpose of this test:</strong> Benchmark the dual-learning ALNS against a representative set of classical combinatorial optimization methods — constructive, local search, evolutionary, and population-based — on a common instance set. Assesses both solution quality (gap to $\\mathrm{LB}_1$) and computational cost (runtime) to determine where the hybrid approach stands in the quality–speed trade-off landscape.</p>\n<p><strong>Scholl-2 · 5 instances · 50 items</strong> — single-run comparison.</p>\n<div class="table-card"><table><thead><tr><th>Method</th><th>Mean gap</th><th>Time (s)</th></tr></thead><tbody><tr><td>FFD / BFD (constructive)</td><td>1.80</td><td>&lt; 0.01</td></tr><tr><td>Simulated Annealing</td><td>1.80</td><td>0.21</td></tr><tr><td>Tabu Search</td><td>1.80</td><td>0.13</td></tr><tr><td>Genetic Algorithm</td><td>1.00</td><td>0.82</td></tr><tr><td><strong>Dual-learning ALNS</strong></td><td><strong>0.20</strong></td><td><strong>0.36</strong></td></tr><tr><td>Ant Colony Optimization</td><td>0.00</td><td>3.03</td></tr></tbody></table></div>\n<p>The dual-learning ALNS achieves a mean gap of <strong>0.20</strong> — the lowest of any tested method except ACO — while running <strong>8× faster</strong> than ACO. No other method in the comparison achieves both a lower gap and a lower runtime.</p>\n<blockquote><p>Stochastic baselines are single-run results; multi-seed averaging may alter relative rankings.</p></blockquote>',
  },
  {
    kind: "dividerContent",
    title: "VI. Synthesis & Conclusion",
    content:
      "<h2>VI. Synthesis &amp; Conclusion</h2>\n<h3>Contributions · Limitations · Future Work</h3>",
  },
  {
    kind: "content",
    title: "VI.1 Synthesis of Results",
    content:
      '<h2>VI.1 Synthesis of Results</h2>\n<div class="table-card"><table><thead><tr><th>Finding</th><th>Evidence</th></tr></thead><tbody><tr><td>Online RL selector is the decisive ML contributor</td><td>Online-only closes gap 0.20 → 0.00; combined does not</td></tr><tr><td>Offline GBT repair: implemented, not yet impactful</td><td>No quality gain; +2× runtime on small instances</td></tr><tr><td>Combined method generalises across families</td><td>≤ 1 bin from $\\mathrm{LB}_1$ on 4 / 5 families</td></tr><tr><td>The two components contribute asymmetrically</td><td>Ablation reveals clear imbalance — clear avenue for improvement</td></tr><tr><td>Dual-learning ALNS dominates classical metaheuristics</td><td>Best gap–runtime trade-off in comparative study</td></tr></tbody></table></div>\n<blockquote><p>All conclusions are bounded by <strong>5-instance evaluation slices</strong> and single-run stochastic baselines. Broader empirical validation is required before general claims can be made.</p></blockquote>',
  },
  {
    kind: "content",
    title: "VI.2 Conclusions",
    content:
      "<h2>VI.2 Conclusions</h2>\n<p><strong>Three contributions:</strong></p>\n<ol><li><strong>Dual-learning ALNS for 1D-BPP</strong> — online and offline ML components with independent toggle switches enabling clean, separable ablation.</li><li><strong>Supervised repair pipeline</strong> — DAgger-lite data augmentation pooling fresh-BFD and post-destruction traces, 11-feature v2 contract, strict quality gates (ROC-AUC, avg. precision), and serialised feature versioning to prevent silent mismatch.</li><li><strong>Warm-start LinUCB bandit</strong> — Beta–Bernoulli Thompson Sampling cold-start mitigation transitioning to disjoint LinUCB after 300 calls, with a gap-normalised reward signal that scales difficulty appropriately.</li></ol>\n<p><strong>Limitations:</strong> small evaluation slices · single-run stochastic baselines · $\\mathrm{LB}_1$ only (weaker than Martello–Toth $L_2$) · GBT overhead not recovered at small scale.</p>\n<p><strong>Future directions:</strong> full benchmark evaluation with averaged runs · upgrade to $L_2$ lower bound · end-to-end RL repair to remove the supervised dependency · cross-family transfer of learned components.</p>",
  },
];

const deck = document.getElementById("deck");

const resultCharts = {
  31: {
    title: "Avg. gap",
    labels: [
      "No learning (baseline)",
      "Online RL only",
      "Offline GBT only",
      "Both combined",
    ],
    values: [0.2, 0.0, 0.2, 0.2],
  },
  32: {
    title: "Mean gap",
    labels: ["Scholl-2", "Falkenauer-T", "Falkenauer-U", "Wäscher", "Hard28"],
    values: [0.2, 1.0, 0.2, 0.6, 0.67],
  },
  33: {
    title: "Mean gap",
    labels: [
      "FFD / BFD (constructive)",
      "Simulated Annealing",
      "Tabu Search",
      "Genetic Algorithm",
      "Dual-learning ALNS",
      "Ant Colony Optimization",
    ],
    values: [1.8, 1.8, 1.8, 1.0, 0.2, 0.0],
  },
};

function el(tag, className, html) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (html !== undefined) node.innerHTML = html;
  return node;
}

function renderSlide(slide, i) {
  const section = el("section", `slide ${slide.kind}`, "");
  section.dataset.slide = String(i + 1).padStart(2, "0");
  section.innerHTML =
    `<div class="markdown-content">${slide.content}</div>` +
    `<footer>Hybrid ALNS for 1D-BPP · ${String(i + 1).padStart(2, "0")} / ${slides.length}</footer>`;
  return section;
}

slides.forEach((slide, i) => deck.appendChild(renderSlide(slide, i)));

function addResultCharts() {
  Object.entries(resultCharts).forEach(([slideNumber, chart]) => {
    const slide = document.querySelector(
      `[data-slide="${String(slideNumber).padStart(2, "0")}"]`,
    );
    const table = slide?.querySelector(".table-card");
    if (!table) return;

    const grid = el("div", "result-grid", "");
    table.before(grid);
    grid.appendChild(table);

    const panel = el(
      "div",
      "result-chart-panel",
      `<h3>${chart.title}</h3><canvas data-result-chart="${slideNumber}" aria-label="${chart.title}" role="img"></canvas>`,
    );
    grid.appendChild(panel);
  });
}

function initResultCharts() {
  if (typeof Chart === "undefined") return;

  document.querySelectorAll("canvas[data-result-chart]").forEach((canvas) => {
    const chart = resultCharts[Number(canvas.dataset.resultChart)];
    if (!chart) return;

    new Chart(canvas, {
      type: "bar",
      data: {
        labels: chart.labels,
        datasets: [
          {
            data: chart.values,
            backgroundColor: "#2c2420",
            borderColor: "#2c2420",
            borderWidth: 1,
          },
        ],
      },
      options: {
        animation: false,
        maintainAspectRatio: false,
        responsive: true,
        plugins: {
          legend: { display: false },
          tooltip: { enabled: true },
        },
        scales: {
          x: {
            ticks: { color: "#5c5650", font: { size: 11 } },
            grid: { display: false },
          },
          y: {
            beginAtZero: true,
            ticks: { display: false },
            grid: { color: "#e2ddd8" },
          },
        },
      },
    });
  });
}

addResultCharts();
// Defer chart initialisation so the browser has completed its first layout
// pass and canvas elements report correct clientWidth / clientHeight.
requestAnimationFrame(() => initResultCharts());

function fitMarkdownContent() {
  document.querySelectorAll(".markdown-content").forEach((content) => {
    content.style.transform = "none";
    content.style.width = "";
    content.style.height = "";

    const scale = Math.min(
      1,
      content.clientWidth / Math.max(content.scrollWidth, 1),
      content.clientHeight / Math.max(content.scrollHeight, 1),
    );

    if (scale < 1) {
      content.style.transform = `scale(${scale})`;
      content.style.width = `${100 / scale}%`;
      content.style.height = `${100 / scale}%`;
    }
  });
}

function fitSlides() {
  if (window.matchMedia("print").matches) return;
  const scale = Math.min(
    (window.innerWidth - 72) / 1600,
    (window.innerHeight - 72) / 900,
    1,
  );
  document.querySelectorAll(".slide").forEach((slide) => {
    slide.style.transform = `scale(${scale})`;
    slide.style.transformOrigin = "top center";
    slide.style.marginBottom = `${900 * (scale - 1)}px`;
  });
  fitMarkdownContent();
}

window.addEventListener("resize", fitSlides);
window.addEventListener("beforeprint", () =>
  document.querySelectorAll(".slide").forEach((slide) => {
    slide.style.transform = "none";
    slide.style.marginBottom = "0";
  }),
);
window.addEventListener("afterprint", fitSlides);
// Run the full fitSlides() (which includes fitMarkdownContent) on "load" so
// that deferred KaTeX scripts have already rendered formulas before we
// measure scrollHeight for content scaling.  The earlier sync call is
// removed; slides are invisible until load anyway in most browsers.
window.addEventListener("load", fitSlides);
