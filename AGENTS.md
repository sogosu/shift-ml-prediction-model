# Repository Guidelines

## Project Goal
Use user-level data from an observation period (e.g., Days 1–3) to predict Day-180 lifetime value (LTV) and measure prediction accuracy against actual Day-180 LTV.

## Project Structure & Modules
- Source: Jupyter notebooks in `ben_work/` (two modeling notebooks).
- Utilities: `add_markdown_explanations.py`, `enhance_markdown_explanations.py` in repo root.
- Docs: `README.md`, this guide, and `CLAUDE.md`.
- Data/artefacts: Not tracked. See `.gitignore` for `data/`, `*.pkl`, `*.csv`, etc.

## Build, Test, and Development
- Create env: `python -m venv .venv && source .venv/bin/activate`
- Install deps (no requirements file):
  `pip install pandas numpy scikit-learn xgboost catboost lightgbm matplotlib seaborn boto3 category-encoders jupyter`
- Run notebooks: `jupyter lab` (or `jupyter notebook`) and execute cells top-to-bottom.
- Optional batch run: consider `papermill` for parameterized runs and reproducibility.

## Coding Style & Naming
- Python style: PEP 8, 4-space indentation, `snake_case` for files/functions, `CapWords` for classes.
- Notebooks: keep small, ordered sections (Import, Load, Feature Eng, Train, Evaluate). Use markdown headings and set `random_state=42` where applicable.
- Formatting/linting: none enforced; if adding Python modules, prefer `black` (line length 88) and `ruff` locally.

## Testing Guidelines
- No formal unit-test suite. Validate by rerunning notebooks end-to-end; compare predicted vs actual D180 LTV.
- If adding Python utilities, place tests under `tests/` and name `test_*.py`; run with `pytest -q`.
- Report metrics: R², RMSE, MAPE, and % within ±10% of actual. Aim for deterministic results (fixed seeds) and document data sample sizes and filtering.

## Commit & Pull Requests
- Commits: concise, imperative subject; include scope, e.g.,
  `feat(notebook): add TF-IDF pipeline for D1–D3`
  `fix(utils): avoid target leakage in encoder`
- PRs must include: clear description, rationale, links to issues, summary of key metrics (screenshots of notebook cells welcome), and any data assumptions.
- Keep diffs focused; do not commit data, credentials, or large artefacts.

## Security & Configuration
- AWS: notebooks expect S3 access via standard credentials (`~/.aws/credentials` or env vars). Never commit secrets; `.env` is ignored.
- Large data and model files are gitignored; store externally (S3) and document paths/versions in PRs.
