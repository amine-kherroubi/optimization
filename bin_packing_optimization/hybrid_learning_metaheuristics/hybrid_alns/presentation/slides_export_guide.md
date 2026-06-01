# Project Structure

```text
presentation/
├── index.html
├── styles.css
├── chart.umd.min.js
├── script.js
└── EXPORT_GUIDE.md
```

# Rendering and Export Guide

## Run locally

```bash
cd /workspace/bin-packing-optimization
python -m http.server 8000 --directory bin_packing_optimization/hybrid_learning_metaheuristics/hybrid_alns/presentation
```

Open `http://localhost:8000` in a modern browser.

## Render slides in a browser

The deck renders 38 fixed-size slides at 1920 × 1080 pixels. The source material is maintained alongside the renderer in `bin_packing_optimization/hybrid_learning_metaheuristics/hybrid_alns/presentation/`.

## Formula handling for Canva

Formula blocks are converted in-browser from KaTeX markup to high-resolution PNG images after the deck loads. This makes equations import into Canva as stable image elements instead of fragile web-font/vector math. Wait a moment after opening the deck before exporting so the PNG conversion can complete.

## Export to PDF

Use the browser print dialog:

1. Open `http://localhost:8000`.
2. Press `Ctrl+P` or `Cmd+P`.
3. Destination: `Save to PDF`.
4. Layout: `Landscape`.
5. Paper size: custom `1920 × 1080 px` when supported, or `20 × 11.25 in` at CSS 96 DPI.
6. Margins: `None`.
7. Scale: `100%`.
8. Background graphics: `On`.
9. Headers and footers: `Off`.
10. Pages per sheet: `1`.

## Recommended free export tools

```bash
chromium --headless --disable-gpu --print-to-pdf=bin-packing-hybrid-alns-canva.pdf --print-to-pdf-no-header --run-all-compositor-stages-before-draw http://localhost:8000
```

```bash
npx decktape generic --size 1920x1080 http://localhost:8000 bin-packing-hybrid-alns-canva.pdf
```

```bash
pdftoppm -png -r 300 bin-packing-hybrid-alns-canva.pdf slide
```

## Canva-compatible export settings

- Aspect ratio: 16:9 widescreen.
- Slide size: 1920 × 1080 px or larger.
- PDF page size: 1920 × 1080 px (`20 × 11.25 in` at CSS 96 DPI).
- Margins: 0.
- Scale: 100%.
- Background graphics: enabled.
- Headers and footers: disabled.
- Export density for PNG: 300 DPI.
- Import into Canva as a PDF to preserve one Canva slide per PDF page.
