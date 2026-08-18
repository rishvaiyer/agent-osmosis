# Cursor build brief — agent-osmosis web UI

Build a polished, marketable web UI for `agent-osmosis`: a landing page, a live interactive
demo, and a research/results section. This doc is the spec — follow it. The Python engine
already exists in `osmosis/`; you build (1) a thin API over it and (2) the frontend.

---

## Goal

A single site that makes a smart visitor think *"this works and these people are rigorous"*
in 30 seconds. Three jobs: **explain** (landing), **prove live** (demo), **show depth**
(research). It must run locally with one command and deploy to Vercel/Netlify.

## Stack

- **Frontend:** Next.js (App Router) + React + TypeScript + Tailwind. Charts: Recharts or
  visx (no heavy deps). No component library bloat — hand-build.
- **Backend:** a thin **FastAPI** service (`server/`) wrapping the existing engine. The
  engine is CPU-only and fast (seconds), so calls can be synchronous. Add CORS for local dev.
- **Do not reimplement the ML.** Import from `osmosis` and call the functions below.

## Design system (match the existing docs/artifacts — this identity is already established)

```
--paper   #F4F7F7   (light bg)      --ink      #15242B
--panel   #FFFFFF                    --ink-soft #47606B
--teal    #0E8F8A   (primary)        --teal-br  #12A9A1
--coral   #D9612E   (accent/warn)    --line     #DCE6E6
Dark mode: paper #0D1518, panel #131F24, ink #DCE7E8, teal #2DD4BF, coral #F08A54, line #223138
```
- **Type:** Spectral (serif display/headings), Public Sans (body), IBM Plex Mono (labels,
  numbers, code). Google Fonts. Tabular-nums for all metrics.
- Support light + dark (respect `prefers-color-scheme`, add a toggle). Teal is the one bold
  color; keep everything else quiet. Generous spacing, ~65ch text column.
- Favicon/emoji: 🧫. Keep it tasteful and technical, not startup-gradient-y.

## API to build (`server/main.py`, FastAPI)

Each endpoint calls the engine and returns JSON. Cache nothing; runs are cheap.

| Method | Route | Body | Returns (calls) |
| --- | --- | --- | --- |
| POST | `/api/warm-vs-cold` | `{source, target, budget_frac, target_acc, trials, seed}` | `warm_vs_cold(...)` → grid, warm_curve, cold_curve, warm_mean, cold_mean, speedup, recipe summary |
| POST | `/api/compounding` | `{target_acc, trials}` | `compounding_chain(...)` → generations[] (quality, warm_mean, size, confidence) |
| POST | `/api/invariance` | `{budget_frac}` | `influence_invariance(task)` → bases, matrix, mean_overlap, random_overlap, verdict |
| POST | `/api/baselines` | `{budget_frac, target_acc, trials}` | `baselines.compare_selection(...)` → rows[] (method, examples_to_target, ci) |
| GET | `/api/recipe/sample` | – | build a task + `extract_recipe`, return the coreset example strings in curriculum order + labels |
| GET | `/api/bases` | – | `list(BASE_MODELS)` for the model pickers |

Imports: `from osmosis import make_task, warm_vs_cold, compounding_chain, influence_invariance, extract_recipe, BASE_MODELS`; `from osmosis import baselines`. Build the task server-side with `make_task(n=600, seed=seed)`.

## Pages

### 1. Landing (`/`)
- **Hero:** headline "Ship the recipe, not the weights." Subhead: one sentence on reusing how
  one model learned to train the next faster. A single big animated stat: **"2.9× less data"**.
  Primary CTA → Demo, secondary → GitHub.
- **The problem** (3 tight cards): every model upgrade = retrain from scratch; it's 15–25% of
  AI budget; the knowledge doesn't carry over.
- **How it works** (4-step horizontal diagram, reuse the pipeline concept): learn → extract
  recipe → warm-start a different model → compound the failures back. Use inline SVG.
- **The honest bit** (a coral-bordered callout): "what transfers is the difficulty structure,
  not the exact influential set" — signal rigor, not hype.
- Footer with links.

### 2. Live demo (`/demo`)
The centerpiece. Left = controls, right = live results.
- Controls: source model + target model (from `/api/bases`), recipe budget slider, target
  accuracy slider, trials. A big **"Run transfer"** button.
- On run: call `/api/warm-vs-cold`, then **animate** the two learning curves drawing in
  (cold crawling, warm rocketing past the target line). Show metric tiles: cold, warm,
  **speedup**, transfer-confidence. A "⚡ ran N models in X.Xs on CPU" line.
- Below: **"Peek inside the recipe"** — a table of the actual coreset sentences in curriculum
  order (from `/api/recipe/sample`), so it's not a black box.
- A second tab/section: **Compounding** — run `/api/compounding`, dual-axis chart (recipe
  quality ↑, warm cost ↓) across generations.

### 3. Research (`/research`)
For the technical reader.
- **Invariance:** run `/api/invariance`, render the Jaccard heatmap + the verdict
  ("invariant, 7× chance" on the linear substrate) with an honest note that on *real*
  transformers it's weak (~1.5×) — link to `docs/technical.md` §8.
- **Baselines:** run `/api/baselines`, horizontal bar chart of examples-to-target with ours
  highlighted; caption that ours beats cold/EL2N/distillation and the coverage-vs-random
  effect is noisy on the toy substrate (honest).
- **The claim, sharpened:** a short prose block: transfer math is crowded; the ownable
  angles are the compounding recipe + the cross-architecture invariance study.
- Link out to the technical explainer and the repo.

## Components to build
`Hero`, `StatCounter` (animated number), `PipelineDiagram` (inline SVG), `ModelPicker`,
`Slider`, `RaceChart` (animated dual-line), `MetricTile`, `RecipeTable`, `CompoundingChart`
(dual-axis), `InvarianceHeatmap`, `BaselineBars`, `HonestCallout`, `ThemeToggle`.

## Acceptance criteria
- `make dev` (or documented commands) starts FastAPI + Next.js locally; demo works end to end.
- Every chart is driven by a real API call to the engine — no hardcoded fake data.
- Light + dark both legible; no horizontal scroll on mobile; charts scroll inside their box.
- Lighthouse ≥ 90 on the landing page. No layout shift on run.
- The animated race reads as a race (cold visibly behind), and the recipe table shows real
  sentences.
- Deploys to Vercel (frontend) with the API as a separate service (Railway/Render/Fly) — do
  not try to run Python inside Vercel edge. Document the two-service deploy.

## Out of scope (don't build)
Auth, accounts, payment, a database, or a recipe marketplace. This is a demo+research site,
not a product yet. Keep it static-fast and stateless; every run recomputes from the engine.

## Reference for tone & content
Read `docs/eli5.md` (plain), `docs/technical.md` (precise), `README.md` (results table), and
`HANDOFF.md` (state). Mirror the honest, no-hype voice — it's the brand.
