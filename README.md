# Cacao

Offline-first cacao disease detection project focused on West African smallholder farmers.

## Repository structure

- `app/` — Flutter Android application for on-device inference and diagnosis UI.
- `ml/` — Dataset preparation, training, export, and validation scripts.
- `cacao_research.md` — Research and landscape analysis.
- `datasets.md` — Dataset notes and references.
- `TODOS.md` — Current roadmap and follow-up tasks.

## Quick start

### Flutter app (`app/`)

```bash
cd app
flutter pub get
flutter test
flutter run --release
```

See `app/README.md` for full setup details.

### ML pipeline (`ml/`)

```bash
cd ml
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run the numbered scripts in order (`00_*.py` → `04_*.py`) based on your pipeline step.

## Project status

The project includes a functional Flutter app scaffold and ML workflow assets, with ongoing validation and field-readiness work tracked in `TODOS.md`.
