# Paper figures

## `figures/model_architecture.{pdf,svg,png}` (Methods, `fig:model_architecture`)

- Source: `figures/src/model_architecture/` — plain HTML + SVG. `lib.js` = drawing helpers (auto-wrapped text with inline `$TeX$` via MathJax → SVG paths, `**bold**`, boxes, arrows, snowflake/flame glyphs); `figure.js` = layout; `export.mjs` = headless Chrome (system install, `playwright-core`) export + layout checks.
- Content is drawn from `docs/methods_revised.tex` only (system as evaluated). Keep symbols identical to the .tex.
- Build: `cd figures/src/model_architecture && npm install && npm run build` → writes the PDF (what LaTeX includes), a standalone SVG with embedded Inter, and a 300 dpi PNG (PNG is git-ignored). `npm run preview` only renders `preview.png` and prints checks.
- Checks printed on every run: text clipped by its box, overlapping text lines, arrows crossing text, min text/math glyph size (must stay ≥ 7 pt), canvas bounds. Verify fonts with `pdffonts figures/model_architecture.pdf` (Inter only, embedded).
- Geometry: width 394 pt = acmsmall `\textwidth`; included at `width=\linewidth`, so sizes in the source are final print sizes. Body 7 pt, titles 8 pt, math 8 pt with scripts clamped to 7 pt. The figure is a flow schematic: one short label per box, with the .tex symbols on the arrows.
- Colours (Okabe–Ito): blue sensor modality, orange language modality, reddish purple LLM calls, greys deterministic code.
