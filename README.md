# Cacao

Offline-first cacao disease detection project focused on West African smallholder farmers.

## Repository structure

- `app/` — Android application for on-device inference and diagnosis UI.
- `ml/` — Dataset preparation, training, export, and validation scripts.
- `cacao_research.md` — Research and landscape analysis.
- `datasets.md` — Dataset notes and references.
- `TODOS.md` — Current roadmap and follow-up tasks.

## Quick start

### Android app (`app/`)

See `app/README.md` for setup, test, and deployment steps.

### ML pipeline (`ml/`)

```bash
cd ml
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run the numbered scripts in order (`00_*.py` → `04_*.py`) based on your pipeline step.

## Project status

The project includes a functional Android app scaffold and ML workflow assets, with ongoing validation and field-readiness work tracked in `TODOS.md`.
