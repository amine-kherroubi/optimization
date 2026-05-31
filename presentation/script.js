const slides = [
  {
    kind: 'title', section: 'Hybrid learning metaheuristics', title: 'Hybrid ALNS for the Bin Packing Problem', kicker: 'One-Dimensional Variant', body: 'A solver combining metaheuristic large-neighborhood search, adaptive bandit-based operator selection, and machine-learned repair via behavioral cloning from BFD.', stats: ['ALNS', 'LinUCB', 'GBT repair', 'BFD expert']
  },
  {
    kind: 'cards', section: 'Introduction', title: 'Introduction', subtitle: '1D-BPP appears wherever fixed-capacity resources must absorb discrete demands at minimum cost.', cards: [
      ['Industrial pressure', 'Cutting stock, vehicle loading, container logistics, and cloud resource allocation all reduce to packing decisions.'],
      ['Classical ceiling', 'Exact branch-and-price can prove optimality, but scales only to a few hundred items.'],
      ['Adaptive opportunity', 'Learning can guide decisions inside the optimizer without replacing the optimization loop.'],
      ['Project goal', 'Embed online operator selection and offline learned repair in independently togglable ALNS components.']
    ], highlight: 'Online learns which destroy operator to use; offline learns which bin should receive each displaced item.'
  },
  {
    kind: 'outline', section: 'Roadmap', title: 'Outline', steps: ['Introduction', 'Problem definition', 'Literature review', 'Proposed solution', 'Tests & results', 'Synthesis & conclusion'], note: 'The deck follows the Markdown slide numbering exactly: 36 slides.'
  },
  {
    kind: 'formula', section: 'I.1 Problem definition', title: 'I.1 Problem Definition', blocks: [
      ['Given', 'Items I = {1,…,n}, integer sizes sᵢ > 0, identical bins of capacity C, and sᵢ ≤ C.'],
      ['Feasibility', 'B₁,…,Bₘ partitions all items and every bin satisfies Σᵢ∈Bⱼ sᵢ ≤ C.'],
      ['Objective', 'minimize m, the number of bins used.']
    ], formula: 'min₍B₁,…,Bₘ₎ m', accent: 'Partition all items · respect capacity · minimize bin count'
  },
  {
    kind: 'formula', section: 'I.2 Lower bound', title: 'I.2 Continuous Relaxation Lower Bound', formula: 'LB₁ = ⌈ Σᵢ₌₁ⁿ sᵢ / C ⌉', blocks: [
      ['Volume argument', 'Any feasible packing must hold total item volume.'],
      ['Capacity argument', 'With m bins, total available capacity is at most mC.'],
      ['Integer result', 'mC ≥ Σsᵢ implies m ≥ Σsᵢ / C, then ceiling rounds to a valid integer lower bound.']
    ], accent: 'Used as a lightweight progress indicator; tighter Martello–Toth L₂ bounds are future work.'
  },
  {
    kind: 'comparison', section: 'I.3 Complexity', title: 'I.3 Complexity and Algorithmic Strategy', columns: ['Approach', 'Representative', 'Guarantee', 'Scalability'], rows: [
      ['Exact', 'Branch-and-bound / price', 'Optimal', 'Few hundred items'],
      ['Approximation', 'FFD / BFD', '≤ 11/9 OPT + O(1)', 'High'],
      ['Metaheuristic', 'ALNS', 'Heuristic', 'High']
    ], callout: 'Strategy: warm start with BFD, then use ALNS to escape dense local optima and push toward LB₁.'
  },
  {
    kind: 'stream', section: 'II. Literature review', title: 'II. Literature Review — ML in Metaheuristics', lanes: [
      ['Adaptive operator selection', 'Fialho et al. (2010), COMPASS, and Maturana & Saubion (2008) motivate bandit-based AOS; this project uses LinUCB (Chu et al. 2011).'],
      ['Learned repair / construction', 'Khalil et al. (2017) shows learned greedy construction; this project uses behavioral cloning to imitate BFD bin assignment.'],
      ['ML-augmented LNS', 'Hottung & Tierney (2020) and Lu et al. (2021) learn richer LNS behavior; this project chooses an interpretable GBT + bandit design.']
    ], callout: 'Positioning: dual-learning ALNS at the intersection of online contextual selection and offline repair ranking.'
  },
  {
    kind: 'divider', section: 'III', title: 'III. Proposed Solution', subtitle: 'BFD warm start · Adaptive LNS · LinUCB operator selection · GBT learned repair'
  },
  {
    kind: 'architecture', section: 'III.0 Global architecture', title: 'III.0 Global Architecture', nodes: ['Instance', 'BFD warm start', 'LinUCB chooses destroy arm', 'Destroy partial solution', 'GBT repair ranks bins', 'SA acceptance', 'Reward update', 'Best solution'], callout: 'Two ML decision points are independently togglable: destroy-operator selection and repair bin ranking.'
  },
  {
    kind: 'process', section: 'III.1 Initialization', title: 'III.1 Constructive Initialization: Best-Fit Decreasing', steps: ['Sort items by non-increasing size', 'Evaluate all open bins that can accommodate the item', 'Place into the feasible bin with minimum post-placement slack', 'Open a new bin only when no feasible bin exists'], callout: 'A stronger warm start reduces recovery time and leaves fewer bins for ALNS to improve.'
  },
  {
    kind: 'divider', section: 'III.2', title: 'III.2 Metaheuristic Framework', subtitle: 'Adaptive Large-Neighborhood Search'
  },
  {
    kind: 'cards', section: 'III.2a Local search failure', title: 'III.2a Why Local Search Alone Fails', subtitle: 'Single-item neighborhoods are efficient, but they cannot reliably cross dense local-optimum barriers.', cards: [
      ['Local move', 'Relocate one item within N(x).'],
      ['Failure mode', 'No single move improves objective even when the solution is globally weak.'],
      ['1D-BPP structure', 'Local optima are tied to specific item groupings across bins.'],
      ['LNS answer', 'Destroy part of the solution, then repair within an implicitly exponential neighborhood.']
    ], highlight: 'Large neighborhoods make escape possible without enumerating every completion.'
  },
  {
    kind: 'flow', section: 'III.2b LNS iterate', title: 'III.2b The LNS Iterate', steps: ['Current solution x', 'Remove displaced set D', 'Partial solution x̂', 'Repair all displaced items', 'Candidate solution x′', 'Adaptive operator update'], callout: 'ALNS extends LNS by adaptively selecting the destroy operator from a portfolio.'
  },
  {
    kind: 'comparison', section: 'III.2c Destroy operators', title: 'III.2c Destroy Operators', columns: ['Arm', 'Operator', 'Mechanism'], rows: [
      ['0', 'Random', 'Uniformly removes min(k,n−1) placed items for broad exploration.'],
      ['1', 'Worst-load', 'Targets low-load bins first to consolidate underused capacity.'],
      ['2', 'Related-item', 'Removes a seed item plus nearest sizes to reshape compatible groupings.']
    ], callout: 'No operator dominates: random helps early exploration; targeted destruction helps near local optima.'
  },
  {
    kind: 'formula', section: 'III.2d Destruction radius', title: 'III.2d Destruction Radius', formula: 'kₘᵢₙ = max(1,⌊0.05n⌋)   ·   kₘₐₓ = max(kₘᵢₙ+1,⌊0.25n⌋)', blocks: [
      ['Default range', 'Each iteration displaces between 5% and 25% of items.'],
      ['Diversification', 'As stagnation grows, kₘₐₓ expands linearly.'],
      ['Intent', 'Broaden the search neighborhood when recent iterations stop improving.']
    ], accent: 'Small enough to preserve structure; large enough to escape local traps.'
  },
  {
    kind: 'divider', section: 'III.3', title: 'III.3 Acceptance Mechanism', subtitle: 'Simulated Annealing with soft reheat and restart'
  },
  {
    kind: 'formula', section: 'III.3a SA acceptance', title: 'III.3a SA Acceptance and Cooling', formula: 'Accept x′ if Δ ≤ 0 or U < exp(−Δ/T)', blocks: [
      ['Cost delta', 'Δ = f(x′) − f(x). Improvements are always accepted.'],
      ['Exploration', 'Worse candidates can enter with probability exp(−Δ/T).'],
      ['Cooling', 'After each iteration: T ← αcool · T, with 0 < αcool ≤ 1.'],
      ['Soft reheat', 'During prolonged stagnation: T ← max(T, 0.35T₀).']
    ], accent: 'Controlled willingness to move uphill prevents premature freezing.'
  },
  {
    kind: 'process', section: 'III.3b Restart', title: 'III.3b Diversification Restart', steps: ['Stagnation reaches no_improve_limit', 'Reset current solution to best solution found so far', 'Reheat temperature to at least 0.20T₀', 'Shrink patience window to trigger future restarts sooner', 'Continue until max_iterations'], callout: 'The restart mechanism diversifies the run; the iteration budget remains the sole hard termination criterion.'
  },
  {
    kind: 'divider', section: 'IV', title: 'IV. Machine Learning Components', subtitle: 'Operator selection + learned repair'
  },
  {
    kind: 'comparison', section: 'IV.0 Components', title: 'IV. Two Independent ML Components', columns: ['Component', 'Decision targeted', 'Method'], rows: [
      ['Contextual bandit', 'Which destroy operator to apply each iteration', 'Warm-start LinUCB'],
      ['Repair ranker', 'Which bin receives each displaced item', 'Behavioral cloning with GBT']
    ], callout: 'Both components can be enabled or disabled separately for clean ablation.'
  },
  {
    kind: 'process', section: 'IV.1 Phase 1', title: 'IV.1 Operator Selection: Phase 1', steps: ['Initialize every arm with Beta(1,1)', 'Sample θ̃ₖ for each destroy arm', 'Select the arm with highest sample', 'Observe reward r and draw a Bernoulli outcome', 'Update only the selected arm'], callout: 'The first 300 calls identify promising operators before LinUCB depends on context features.'
  },
  {
    kind: 'formula', section: 'IV.1 Phase 2', title: 'IV.1 Operator Selection: Phase 2', formula: 'k* = argmaxₖ [ θ̂ₖᵀx + α √(xᵀAₖ⁻¹x) ]', blocks: [
      ['State', 'Each arm maintains Aₖ⁻¹ ∈ R⁵ˣ⁵ and bₖ ∈ R⁵.'],
      ['Exploitation', 'Calls 301+ use disjoint LinUCB with α = 0.3, following Chu et al. (2011).'],
      ['Update', 'Sherman–Morrison rank-1 update keeps each step O(d²).']
    ], accent: 'Calls 301+ transition from cold-start exploration to context-aware selection.'
  },
  {
    kind: 'comparison', section: 'IV.1 Context vector', title: 'IV.1 Context Vector', columns: ['Index', 'Feature', 'Meaning'], rows: [
      ['0', 'T / T₀', 'Temperature: hot to cold'],
      ['1', 'iter_no_improve / no_improve_limit', 'Progress toward restart'],
      ['2', 'LB₁ / f(x)', 'Lower-bound-to-cost ratio'],
      ['3', 'k / n', 'Current destruction radius'],
      ['4', 'iteration / max_iterations', 'Overall search progress']
    ], callout: 'All features lie in [0,1], keeping the UCB exploration term comparable across dimensions.'
  },
  {
    kind: 'formula', section: 'IV.1 Reward signal', title: 'IV.1 Reward Signal', formula: 'r = saved bins / current gap   ·   0.2 if accepted without saving   ·   0 if rejected', blocks: [
      ['Gap', 'gap = max(1, f(x) − LB₁) after incumbent update.'],
      ['Savings', 'Bin reductions earn up to 1.0, normalized by remaining difficulty.'],
      ['Accepted neutral moves', 'Accepted candidates without bin savings receive 0.2.'],
      ['Rejected moves', 'No accepted improvement path means reward 0.0.']
    ], accent: 'Near the lower bound, each additional saved bin becomes more valuable.'
  },
  {
    kind: 'cards', section: 'IV.2 Repair overview', title: 'IV.2 Machine-Learned Repair: Overview', subtitle: 'After destruction, each displaced item is reinserted in non-increasing size order.', cards: [
      ['Feasible set', 'F(i) = {j : loadⱼ + sᵢ ≤ C}.'],
      ['No feasible bin', 'Open a new bin immediately.'],
      ['Feasible bins exist', 'Score every candidate bin and place into the highest-scoring one.'],
      ['Training objective', 'Binary classifier: label 1 for the bin BFD would choose, 0 otherwise.']
    ], highlight: 'Behavioral cloning turns BFD into a fast repair policy for partial ALNS solutions.'
  },
  {
    kind: 'comparison', section: 'IV.2 Features', title: 'IV.2 Feature Representation', columns: ['Group', 'Features', 'Purpose'], rows: [
      ['Item', 'sᵢ/C, (sᵢ/C)², rank(i)/n, remaining/n', 'Capture size, priority, and repair progress.'],
      ['Bin load', 'ℓⱼ/C, (C−ℓⱼ)/C, post-placement slack', 'Measure current and future capacity fit.'],
      ['Bin contents', '|Bⱼ|/n, max size/C, min size/C', 'Summarize bin composition.'],
      ['Compatibility', 'sᵢ/(C−ℓⱼ)', 'Estimate how much residual capacity the item consumes.']
    ], callout: 'Every feature is normalized by capacity or item count to improve cross-instance transfer.'
  },
  {
    kind: 'cards', section: 'IV.2 Model training', title: 'IV.2 Model Architecture and Training Data', subtitle: 'The model optimizes ranking probabilities, not hard class decisions.', cards: [
      ['Model', 'GradientBoostingClassifier with StandardScaler serialized alongside the bundle.'],
      ['Fast inference', 'NumPy scaler plus raw GBT prediction bypasses per-call validation.'],
      ['Quality gates', 'Feature version and n_features are validated at load time; ROC-AUC is the primary metric.'],
      ['Trace mix', 'Fresh BFD traces plus post-destruction traces reduce distribution shift.']
    ], highlight: 'Balanced sample weights and BFD labels turn feasible (item, bin) pairs into ranked repair candidates.'
  },
  {
    kind: 'divider', section: 'V', title: 'V. Experimental Evaluation', subtitle: 'Protocol · benchmarks · ablation · comparative results'
  },
  {
    kind: 'comparison', section: 'V.1 Protocol', title: 'V.1 Experimental Protocol', columns: ['Setting', 'Value'], rows: [
      ['Random seed', '42'], ['ALNS iterations', '300'], ['Initial temperature T₀', '1 / ln 2'], ['Cooling coefficient αcool', '0.9995'], ['LinUCB exploration α', '0.3'], ['Warm-up', '300 Thompson-sampling calls'], ['Repair model', 'Single pre-trained bundle across datasets'], ['Environment', 'Python 3.13 · scikit-learn · Windows 11 · i9-13950HX · 64 GB RAM']
    ], callout: 'Primary metric: gap = bins used − LB₁. Gap 0 means LB-certifiably optimal.'
  },
  {
    kind: 'comparison', section: 'V.2 Benchmarks', title: 'V.2 Benchmark Datasets — Instance Characteristics', columns: ['Family', 'Capacity', 'Items', 'Structure'], rows: [
      ['Scholl-2', '1,000', '50–500', 'Uniform sizes'], ['Falkenauer-T', '1,000', '—', 'Triplet structure'], ['Falkenauer-U', '150', '—', 'Uniform sizes'], ['Wäscher', '10,000', '—', 'Cutting-stock'], ['Hard28', '1,000', '160–200', 'Adversarially hard']
    ], callout: 'Falkenauer-T triplets systematically weaken LB₁, so its reported gaps are not directly comparable.'
  },
  {
    kind: 'chart', section: 'V.3 Test 1', title: 'V.3 Test 1 — Ablation Study', subtitle: 'Scholl-2 · 5 instances · 50 items · independently toggled components', labels: ['Baseline', 'Online RL', 'Offline GBT', 'Both'], values: [0.2, 0.0, 0.2, 0.2], metrics: [['Configuration', 'Avg. bins', 'Avg. gap', 'Fill %', 'Time (s)'], ['No learning', '18.2', '0.20', '92.76', '0.153'], ['Online RL only', '18.0', '0.00', '93.76', '0.186'], ['Offline GBT only', '18.2', '0.20', '92.76', '0.353'], ['Both combined', '18.2', '0.20', '92.76', '0.362']], callout: 'Online RL closes the mean gap to 0 alone. Offline GBT adds roughly 2× runtime without quality gain on this small slice.'
  },
  {
    kind: 'chart', section: 'V.4 Test 2', title: 'V.4 Test 2 — Multi-Dataset Generalization', subtitle: 'Combined method across all benchmark families', labels: ['Scholl-2', 'Falk-T', 'Falk-U', 'Wäscher', 'Hard28'], values: [0.20, 1.00, 0.20, 0.60, 0.67], metrics: [['Family', 'Mean gap', 'Runtime (s)'], ['Scholl-2', '0.20', '0.6–1.2'], ['Falkenauer-T', '1.00', '1.0–1.7'], ['Falkenauer-U', '0.20', '4.3–5.8'], ['Wäscher', '0.60', '1.0–2.2'], ['Hard28', '0.67', '13.5–15.1']], callout: '4 of 5 families land within 1 bin of LB₁; Falkenauer-T likely reflects the weakness of LB₁ on triplet instances.'
  },
  {
    kind: 'chart', section: 'V.5 Test 3', title: 'V.5 Test 3 — Comparative Study', subtitle: 'Scholl-2 · 5 instances · 50 items · single-run comparison', labels: ['FFD/BFD', 'SA', 'Tabu', 'GA', 'Dual ALNS', 'ACO'], values: [1.8, 1.8, 1.8, 1.0, 0.2, 0.0], metrics: [['Method', 'Mean gap', 'Time (s)'], ['FFD / BFD', '1.80', '< 0.01'], ['Simulated Annealing', '1.80', '0.21'], ['Tabu Search', '1.80', '0.13'], ['Genetic Algorithm', '1.00', '0.82'], ['Dual-learning ALNS', '0.20', '0.36'], ['Ant Colony Optimization', '0.00', '3.03']], callout: 'Dual-learning ALNS is second-best in gap, runs about 8× faster than ACO, and dominates classical metaheuristics in this slice.'
  },
  {
    kind: 'divider', section: 'VI', title: 'VI. Synthesis & Conclusion', subtitle: 'Contributions · limitations · future work'
  },
  {
    kind: 'comparison', section: 'VI.1 Synthesis', title: 'VI.1 Synthesis of Results', columns: ['Finding', 'Evidence'], rows: [
      ['Online RL selector is decisive', 'Online-only closes gap 0.20 → 0.00.'],
      ['Offline GBT not yet impactful', 'No quality gain; +2× runtime on small instances.'],
      ['Combined method generalizes', '≤ 1 bin from LB₁ on 4 / 5 families.'],
      ['Components contribute asymmetrically', 'Ablation exposes clear improvement path.'],
      ['Hybrid ALNS is competitive', 'Best gap–runtime trade-off against classical metaheuristics.']
    ], callout: 'Conclusions are bounded by 5-instance slices and single-run stochastic baselines.'
  },
  {
    kind: 'final', section: 'VI.2 Conclusions', title: 'VI.2 Conclusions', contributions: ['Online + offline ML components with independent toggles for clean ablation.', 'Supervised repair pipeline with DAgger-lite traces, 11-feature contract, quality gates, and versioning.', 'Warm-start LinUCB with Thompson-sampling cold start and gap-normalized rewards.'], limitations: ['Small evaluation slices', 'Single-run stochastic baselines', 'LB₁ only', 'GBT overhead not recovered at small scale'], future: ['Full benchmark evaluation with averaged runs', 'Upgrade to L₂ lower bound', 'End-to-end RL repair', 'Cross-family transfer of learned components']
  }
];

const deck = document.getElementById('deck');

function el(tag, className, html) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (html !== undefined) node.innerHTML = html;
  return node;
}

function header(slide, i) {
  return `<div class="topline"><span>${String(i + 1).padStart(2, '0')}</span><b>${slide.section}</b></div><h2>${slide.title}</h2>${slide.subtitle ? `<p class="subtitle">${slide.subtitle}</p>` : ''}`;
}

function renderCards(slide) {
  return `${header(slide, slides.indexOf(slide))}<div class="card-grid">${slide.cards.map((c, i) => `<article class="info-card tone-${i % 4}"><span>${String(i + 1).padStart(2, '0')}</span><h3>${c[0]}</h3><p>${c[1]}</p></article>`).join('')}</div>${slide.highlight ? `<div class="callout">${slide.highlight}</div>` : ''}`;
}

function renderComparison(slide) {
  return `${header(slide, slides.indexOf(slide))}<div class="table-card"><table><thead><tr>${slide.columns.map(c => `<th>${c}</th>`).join('')}</tr></thead><tbody>${slide.rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>${slide.callout ? `<div class="callout">${slide.callout}</div>` : ''}`;
}

function renderFormula(slide) {
  return `${header(slide, slides.indexOf(slide))}<div class="formula-layout"><div class="formula-card">${slide.formula}</div><div class="mini-stack">${slide.blocks.map((b, i) => `<article><span>${String(i + 1).padStart(2, '0')}</span><h3>${b[0]}</h3><p>${b[1]}</p></article>`).join('')}</div></div>${slide.accent ? `<div class="callout">${slide.accent}</div>` : ''}`;
}

function renderProcess(slide) {
  return `${header(slide, slides.indexOf(slide))}<div class="process-track">${slide.steps.map((s, i) => `<article><span>${String(i + 1).padStart(2, '0')}</span><p>${s}</p></article>`).join('')}</div>${slide.callout ? `<div class="callout">${slide.callout}</div>` : ''}`;
}

function renderFlow(slide) {
  return `${header(slide, slides.indexOf(slide))}<div class="flow-row">${slide.steps.map((s, i) => `<article><span>${String(i + 1).padStart(2, '0')}</span><p>${s}</p></article>`).join('')}</div>${slide.callout ? `<div class="callout">${slide.callout}</div>` : ''}`;
}

function renderChart(slide) {
  const idx = slides.indexOf(slide);
  const [head, ...rows] = slide.metrics;
  return `${header(slide, idx)}<div class="chart-layout"><div class="chart-panel"><h3>Mean gap</h3><canvas id="gap-chart-${idx}" data-chart-index="${idx}" data-chart-kind="gap"></canvas></div><div class="chart-panel"><h3>Source values</h3><table class="metric-table"><thead><tr>${head.map(c => `<th>${c}</th>`).join('')}</tr></thead><tbody>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div></div><div class="callout">${slide.callout}</div>`;
}

function renderSlide(slide, i) {
  const section = el('section', `slide ${slide.kind}`, '');
  section.dataset.slide = String(i + 1).padStart(2, '0');
  let html = '';
  if (slide.kind === 'title') html = `<div class="hero"><div><p class="eyebrow">${slide.section}</p><h1>${slide.title}</h1><h3>${slide.kicker}</h3><p>${slide.body}</p><div class="pill-row">${slide.stats.map(s => `<span>${s}</span>`).join('')}</div></div><div class="hero-visual"><div class="bin big"><i style="height:34%"></i><i style="height:26%"></i><i style="height:21%"></i></div><div class="orbit"><span></span><span></span><span></span></div></div></div>`;
  if (slide.kind === 'cards') html = renderCards(slide);
  if (slide.kind === 'outline') html = `${header(slide, i)}<div class="outline-list">${slide.steps.map((s, idx) => `<article><b>${String(idx + 1).padStart(2, '0')}</b><span>${s}</span></article>`).join('')}</div><div class="callout">${slide.note}</div>`;
  if (slide.kind === 'formula') html = renderFormula(slide);
  if (slide.kind === 'comparison') html = renderComparison(slide);
  if (slide.kind === 'stream') html = `${header(slide, i)}<div class="stream-lanes">${slide.lanes.map((l, idx) => `<article><b>Stream ${idx + 1}</b><h3>${l[0]}</h3><p>${l[1]}</p></article>`).join('')}</div><div class="callout">${slide.callout}</div>`;
  if (slide.kind === 'divider') html = `<div class="divider-inner"><p class="eyebrow">${slide.section}</p><h1>${slide.title}</h1><h3>${slide.subtitle}</h3></div>`;
  if (slide.kind === 'architecture') html = `${header(slide, i)}<div class="arch"><svg viewBox="0 0 1200 360" aria-hidden="true"><path d="M135 180H1065"/><path d="M275 180c55-90 135-90 190 0s135 90 190 0 135-90 190 0"/></svg>${slide.nodes.map((n, idx) => `<article class="arch-node n${idx}"><b>${String(idx + 1).padStart(2, '0')}</b><span>${n}</span></article>`).join('')}</div><div class="callout">${slide.callout}</div>`;
  if (slide.kind === 'process') html = renderProcess(slide);
  if (slide.kind === 'flow') html = renderFlow(slide);
  if (slide.kind === 'chart') html = renderChart(slide);
  if (slide.kind === 'final') html = `${header(slide, i)}<div class="final-grid"><article><h3>Contributions</h3>${slide.contributions.map(x => `<p>${x}</p>`).join('')}</article><article><h3>Limitations</h3><div class="tag-cloud">${slide.limitations.map(x => `<span>${x}</span>`).join('')}</div></article><article><h3>Future directions</h3><div class="tag-cloud">${slide.future.map(x => `<span>${x}</span>`).join('')}</div></article></div>`;
  section.innerHTML = html + `<footer>Hybrid ALNS for 1D-BPP · ${String(i + 1).padStart(2, '0')} / ${slides.length}</footer>`;
  return section;
}

slides.forEach((slide, i) => deck.appendChild(renderSlide(slide, i)));

function fitSlides() {
  if (window.matchMedia('print').matches) return;
  const scale = Math.min((window.innerWidth - 72) / 1600, (window.innerHeight - 72) / 900, 1);
  document.querySelectorAll('.slide').forEach(slide => {
    slide.style.transform = `scale(${scale})`;
    slide.style.transformOrigin = 'top center';
    slide.style.marginBottom = `${900 * (scale - 1)}px`;
  });
}

window.addEventListener('resize', fitSlides);
window.addEventListener('beforeprint', () => document.querySelectorAll('.slide').forEach(slide => {
  slide.style.transform = 'none';
  slide.style.marginBottom = '0';
}));
window.addEventListener('afterprint', fitSlides);
fitSlides();


function initCharts() {
  if (typeof Chart === 'undefined') return;
  document.querySelectorAll('canvas[data-chart-index]').forEach(canvas => {
    const slide = slides[Number(canvas.dataset.chartIndex)];
    new Chart(canvas, {
      type: 'bar',
      data: {
        labels: slide.labels,
        datasets: [{
          label: 'Mean gap',
          data: slide.values,
          backgroundColor: ['#2563eb', '#06b6d4']
        }]
      },
      options: { plugins: { title: { text: 'Lower is better' } } }
    });
  });
}

requestAnimationFrame(initCharts);
