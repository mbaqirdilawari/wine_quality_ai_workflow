# Wine Quality Analysis: Project Plan

**Question:** Can a few basic chemical measurements (like alcohol content) give us a good idea of a wine's quality score?

**Dataset:** `data/wine_quality_merged.csv`: 6,497 red and white wines, 11 chemical measurements, a `type` column (`red`/`white`) and an integer `quality` score (3–9).

**Workflow:** Architect → Builder → Tester, each in a fresh chat. This file is the single living plan. It is owned by the Architect, and stages are appended, never overwritten. The Builder and Tester may only correct the Manual Smoke Test wording of the current stage if commands drift.

---

## Stage 1: Wine Quality rebuild

### Goal
Rebuild the analysis as a small, modular Python project. The pipeline is load → inspect → clean → explore → model → charts. It runs the same way locally and in Docker, through a Makefile, and has tests written alongside the code.

### Requirements
- **Load and inspect:** print the shape, missing values per column, and the count of exact duplicate rows.
  - Observed baseline: 6,497 × 13, 0 missing, 1,177 exact duplicates.
- **Clean early:** remove exact duplicate rows (expect 5,320 rows: 3,961 white, 1,359 red). Keep outliers. Cleaning happens right after loading, before any exploration or modelling.
  - Why keep outliers: they are real, lab-measured wines, not entry errors. Removing them would bias the model toward "average" wines and drop most of the rare quality-3 and quality-9 wines.
  - Why remove duplicates: identical rows over-weight some wines, and they can land in both the train and test splits, which inflates test scores.
  - Both reasons are printed at runtime and written in the README.
- **Explore (light):**
  - Filters, for example: wines with alcohol ≥ 12%, only red wines, and only high-quality wines (quality ≥ 7).
  - Group summaries: the mean of key features by `type`, and the count plus mean alcohol by `quality`.
- **Model:** a linear regression that predicts `quality`, comparing two feature sets:
  - 3 features: `alcohol`, `volatile acidity`, `sulphates`
  - all 11 chemical features (`type` excluded)
  - Report R² and RMSE on a held-out test set (80/20 split, `random_state=42`) and print a short comparison.
- **Charts (PNG):**
  - `alcohol_by_quality_boxplot.png`
  - `alcohol_vs_density_scatter.png`, with a linear trend line
- **Configurable paths** through environment variables with defaults:
  - `WINE_DATA_PATH` (default `data/wine_quality_merged.csv`)
  - `WINE_OUTPUT_DIR` (default `outputs/`, created if missing)
- **Quality gates:** pytest (typical and edge cases), black, flake8.
- **Interface:** a Makefile with `install`, `run`, `test`, `lint`, `format`, `docker-build`, `docker-run` (plus `docker-test`, `compose-up`, `compose-test` and `clean`).
- **Optional Docker Compose:** a small `docker-compose.yml` lets you run the analysis with `docker compose up` and the tests with `docker compose run --rm tests`. It is an extra option alongside `make docker-build` and `make docker-run`, not a replacement for them.
- **Isolated environment:** `make install` creates a project-local `.venv` (Python 3.13) and installs the requirements into it. `run`, `test`, `lint` and `format` all use that `.venv`. Nothing is installed into the base environment.
- **Python version:** 3.13 both locally and in the container (`python:3.13-slim`).
- **Documentation:** a Dockerfile, a `.dockerignore`, and a concise README that also covers the assignment's reflection items (see Build order step 10).

### Proposed changes
Proposed project structure:

```
wine_quality_ai_workflow/
├── data/wine_quality_merged.csv   # input (exists; not modified)
├── docs/
│   ├── plan.md                    # this plan
│   └── screenshots/               # docker-build and docker-run screenshots for the README (added by the user)
├── .venv/                         # created by `make install` (git-ignored, docker-ignored)
├── wine_analysis/                 # the Python package (one job per module)
│   ├── __init__.py
│   ├── config.py      # get_data_path(), get_output_dir(): read env vars with defaults
│   ├── data.py        # load_data(path), inspect_data(df) -> dict, clean_data(df) -> DataFrame
│   ├── explore.py     # filter_* helpers, summary_by_type(df), summary_by_quality(df)
│   ├── model.py       # FEATURES_BASIC, FEATURES_ALL, train_and_evaluate(df, features) -> dict
│   ├── plots.py       # plot_alcohol_by_quality(df, out_dir), plot_alcohol_vs_density(df, out_dir) -> Path
│   └── main.py        # orchestrates the pipeline and prints results; `python -m wine_analysis.main`
├── tests/
│   ├── conftest.py    # small synthetic DataFrame fixtures
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_explore.py
│   ├── test_model.py
│   ├── test_plots.py
│   └── test_pipeline.py   # regression and integration tests on the real CSV
├── outputs/           # generated charts (git-ignored; .gitkeep only)
├── requirements.txt   # pandas, numpy, scikit-learn, matplotlib, pytest, black, flake8
├── setup.cfg          # flake8 config (max-line-length = 88, matches black)
├── Makefile           # public interface; local targets run through .venv
├── Dockerfile
├── docker-compose.yml # optional: `analysis` service, plus a `tests` service (profile "test")
├── .dockerignore
├── .gitignore
└── README.md
```

Interface notes for the Builder:
- `load_data(path)`
  - Raises `FileNotFoundError` with a clear message if the file is missing.
  - Raises `ValueError` if any expected column is missing.
- `inspect_data(df)` returns `{"shape": (rows, cols), "missing": {col: n}, "duplicates": n}`. It is pure: it returns data and never prints.
- `clean_data(df)` returns a new DataFrame with exact duplicates dropped and the index reset. It never mutates its input and does not touch outliers.
- `train_and_evaluate(df, features, test_size=0.2, random_state=42)` returns `{"features": [...], "r2": float, "rmse": float, "n_train": int, "n_test": int}`. It raises `ValueError` for unknown feature names or too few rows.
- The plot functions return the saved file path and create `out_dir` if it does not exist.
- `main.main()` is the only place that prints and the only place that reads config.

### Architecture / boundaries
- **One responsibility per module.** Logic modules return data (DataFrames and dicts) and `main.py` does the printing. This keeps the functions easy to test and avoids the "one big file" problem from the previous version.
- **Clean first.** `main` calls `clean_data` immediately after `load_data`, and every later step receives the cleaned frame.
- **No side effects at import time.** Environment variables are read inside the `config` functions, not at module level, so tests can use `monkeypatch`.
- **Headless plotting.** Use the matplotlib `Agg` backend, close each figure after saving, and use matplotlib only (no seaborn) to keep dependencies minimal.
- **Column names contain spaces** (`volatile acidity`). Define them once as constants in `model.py` and `data.py` and use them as written. Do not rename columns.
- **Flat package, no `src/` layout and no `pyproject` packaging.** Run code with `python -m wine_analysis.main` and tests with `python -m pytest` from the repo root, so imports work without installing the package.
- **Deterministic results.** The fixed `random_state=42` makes the reported metrics reproducible.
- **Makefile and `.venv`:**
  - The Makefile defines `PYTHON ?= python3.13` and `VENV_PY := .venv/bin/python` at the top.
  - `make install` runs `$(PYTHON) -m venv .venv`, upgrades pip with `$(VENV_PY) -m pip install --upgrade pip`, then runs `$(VENV_PY) -m pip install -r requirements.txt`.
  - `run`, `test`, `lint` and `format` call the venv's Python directly, never `python`, `pytest` or `black` from the shell:
    - `$(VENV_PY) -m wine_analysis.main`
    - `$(VENV_PY) -m pytest`
    - `$(VENV_PY) -m black`
    - `$(VENV_PY) -m flake8`
  - This means you don't have to activate the venv, and the base environment is never used.
  - Calling tools through `python -m` also avoids broken script shebangs caused by the spaces in the project path.
  - `make clean` removes caches and generated charts, but not `.venv`. Delete `.venv` by hand if you want a fresh install.
- **The container does not use a venv.** The image installs the requirements into its own system Python, because the container is already an isolated environment. The Docker targets therefore don't touch `.venv`.

### Risks and design concerns
- **Weak predictive power is expected.** Quality is a subjective, mostly 5–6 score. A linear model will likely reach an R² of only about 0.2–0.3. That is a valid answer to the question ("only partly"), not a bug. The README should say so honestly, and the tests should not assert high accuracy.
- **Treating quality as continuous.** Linear regression treats an ordinal score as continuous, so predictions can come out as values like 5.7. This is acceptable for a beginner course and is noted in the README.
- **Class imbalance.** Quality 9 has only 5 wines and quality 3 has only 30. The boxplot will show thin groups for these scores. Keep them and do not merge classes.
- **Duplicate leakage.** Splitting before deduplicating would leak identical rows across train and test. This is why cleaning comes first.
- **Spaces in the repo path** (`Fall 2026/IDS 706 - Data Engineering/...`). Every path in the Makefile, and especially `-v "$(CURDIR)/outputs:/app/outputs"`, must be quoted, or the Docker mounts break.
- **OneDrive-synced folder.** Docker Desktop must be allowed to share this path. If files are "online-only", the CSV must be downloaded locally first. Generated files in `outputs/` will sync to OneDrive. This is harmless but is one more reason to git-ignore them.
- **Python version alignment.** Both the local `.venv` and the image use Python 3.13 (`python:3.13-slim`), so behaviour and pinned versions match. Two things remain to watch:
  - Pin package versions that publish Python 3.13 wheels for both macOS and Linux (recent pandas, numpy, scikit-learn and matplotlib all do), so neither environment has to compile from source.
  - `make install` must create the venv with Python 3.13. If `python3.13` is not on your PATH, override it with `make install PYTHON=/path/to/python3.13`. Running `.venv/bin/python --version` confirms which version was used.
- **Paths with spaces and `.venv`.** Console-script shebangs inside a venv can break when the absolute path contains spaces. Avoid this by always calling tools as `.venv/bin/python -m <tool>`, as described in Architecture.
- **Regression tests depend on the real CSV.** If the data file changes, the regression numbers (1,177 duplicates and 5,320 rows) change too. This is intentional: they act as data-drift alarms.

### Containerization
- **Base image:** `python:3.13-slim`, which matches the local Python version. Set `WORKDIR /app`, run `pip install --no-cache-dir -r requirements.txt` (copied first so the layer is cached), then copy `wine_analysis/`, `tests/`, `data/`, `setup.cfg`.
- **What it runs:** the default `CMD ["python", "-m", "wine_analysis.main"]` runs the full analysis. Tests run in the same image by overriding the command, `docker run --rm wine-quality python -m pytest -q`, which is what `make docker-test` does.
- **Defaults inside the image:** `ENV WINE_DATA_PATH=/app/data/wine_quality_merged.csv WINE_OUTPUT_DIR=/app/outputs`. The data is baked in, so a plain `docker run` works.
- **How the charts get out:** `make docker-run` bind-mounts the host's `outputs/` folder onto `/app/outputs`:
  `docker run --rm -v "$(CURDIR)/outputs:/app/outputs" wine-quality`
  The PNGs written in the container appear in `./outputs` on the host.
  - To use different data, mount it and point the environment variable at it, for example `-v "/some/dir:/input:ro" -e WINE_DATA_PATH=/input/file.csv`. This is documented in the README.
- **`.dockerignore`:** `.git`, `.venv/`, `outputs/`, `__pycache__/`, `.pytest_cache/`, `*.pyc`, `docs/`, `.DS_Store`. Excluding `.venv/` matters: it contains macOS binaries that would bloat the build context and do not work on Linux.
- **Docker Compose is optional.** The project has a single container and doesn't need Compose, but it is included as a one-command convenience and as practice. The `make docker-*` targets stay the primary path and keep working unchanged. `docker-compose.yml` contains:

  ```yaml
  services:
    analysis:
      build: .
      image: wine-quality            # same image name as make docker-build
      environment:
        WINE_OUTPUT_DIR: /app/outputs
      volumes:
        - ./outputs:/app/outputs     # charts appear in ./outputs on the host
    tests:
      build: .
      image: wine-quality
      command: ["python", "-m", "pytest", "-q"]
      profiles: ["test"]             # not started by `docker compose up`
  ```

  - `docker compose up --build` runs only `analysis`, because `tests` sits behind a profile. Its logs print to the terminal, and the container exits when the analysis finishes.
  - `docker compose run --rm tests` runs the test suite in the same image. Targeting a profiled service with `run` enables it automatically, so no extra flags are needed.
  - Afterwards, `docker compose down` removes the stopped `analysis` container.
  - **Keep the Compose file and the Makefile in sync.** Both use the image name `wine-quality`, the same `/app/outputs` mount, and the same defaults. If one changes, change the other.
  - **No `version:` key.** It is obsolete in Compose v2 and only triggers a warning.
  - **No `container_name`.** Leaving it out avoids name clashes between runs.
- **Makefile Docker targets:**

  | Target | Command | Notes |
  |---|---|---|
  | `docker-build` | `docker build -t wine-quality .` | unchanged |
  | `docker-run` | `docker run --rm -v "$(CURDIR)/outputs:/app/outputs" wine-quality` | unchanged |
  | `docker-test` | `docker run --rm wine-quality python -m pytest -q` | unchanged |
  | `compose-up` | `docker compose up --build && docker compose down` | runs the analysis, then cleans up |
  | `compose-test` | `docker compose run --rm --build tests` | runs the tests through Compose |

### Build order
1. **Setup:**
   - Create `requirements.txt`, `setup.cfg`, `outputs/.gitkeep`, the empty package and the tests folder.
   - Create the `Makefile` with the `.venv`-based install/run/test/lint/format/clean targets.
   - Create `.gitignore`, which must include `.venv/` and `outputs/*.png`.
   - Run `make install` and confirm that `.venv/bin/python --version` reports 3.13.
2. **Config:** `config.py` plus `test_config.py` (defaults and env overrides).
3. **Load, inspect, clean:** `data.py` plus `test_data.py`.
4. **Exploration:** `explore.py` plus `test_explore.py`.
5. **Model:** `model.py` plus `test_model.py`.
6. **Charts:** `plots.py` plus `test_plots.py`.
7. **Orchestration:** `main.py` with clearly headed printed sections, plus `test_pipeline.py` (regression and integration).
8. **Quality pass:** `make format && make lint && make test`, all green.
9. **Docker:**
   - Write `Dockerfile`, `.dockerignore`, and the Makefile targets `docker-build`, `docker-run`, `docker-test`.
   - Verify that the charts appear in `./outputs` and that the tests pass in the container.
   - Only once that works, add `docker-compose.yml` and the `compose-up` and `compose-test` targets, and verify that both produce the same results.
10. **README:** keep it concise and use these sections in order. `[USER]` marks a placeholder that only the user can fill in; the Builder writes the placeholder text as an HTML comment or an obvious `> TODO (user): ...` line, never with invented content.
    1. **Project and question:** a short description, the dataset, and the question being answered.
    2. **Assignment option:** "Option 2: rebuilding my previous Wine Quality project using an Architect → Builder → Tester AI workflow." The Builder writes this.
    3. **Setup and usage:**
       - `make install` (creates `.venv`), `make run`, `make test`, `make lint`, `make format`
       - `make docker-build`, `make docker-run`, `make docker-test`
       - optionally `docker compose up --build` and `docker compose run --rm tests` (or `make compose-up` and `make compose-test`)
       - the environment variables `WINE_DATA_PATH` and `WINE_OUTPUT_DIR`, with an example of mounting other data
    4. **Project structure:** the file tree with a one-line purpose for each file.
    5. **Data cleaning decisions:** why duplicates are removed and outliers are kept.
    6. **Results:** the R² and RMSE table for 3 vs 11 features, the two embedded charts from `outputs/`, a short answer to the question, and the limitations (ordinal target, class imbalance).
    7. **Docker:** what the container runs, how the charts get out through the bind mount, and a short "Optional: Docker Compose" subsection. That subsection explains what the two services do, why the project doesn't strictly need Compose (one container, no other services), and that it is included for convenience. It is followed by two screenshots:
       - `docs/screenshots/docker-build.png`, showing `make docker-build` succeeding
       - `docs/screenshots/docker-run.png`, showing `make docker-run` output and the charts in `outputs/`
       - `[USER]` captures both screenshots. The Builder adds the image links with a placeholder note.
    8. **Manual smoke test result:** the commands that were run, as listed in this plan. `[USER]` records the actual outcome (pass/fail, the key printed numbers, any issues and fixes).
    9. **AI workflow:** how each role contributed.
       - The Builder drafts a factual outline of what each role produced:
         - Architect → `docs/plan.md`
         - Builder → code, Makefile and Docker
         - Tester → tests and verification
       - `[USER]` adds their own description and judgement.
    10. **AI recommendations accepted / changed:** a placeholder for at least one accepted and one changed or rejected recommendation. `[USER]` fills it in. Suggested candidates from this stage:
        - accepted: matplotlib only, clean before exploring
        - changed: the Architect proposed `python:3.12-slim` and installing into the active environment; the user changed these to `python:3.13-slim` and a project `.venv`
        - changed: the Architect said Compose was not needed; the user added it as an optional extra for convenience and learning, and kept the `make docker-*` targets as the primary path
    11. **Independent verification:** `[USER]` describes how they checked the result themselves, for example re-running the smoke test, comparing the printed counts (6497 / 1177 / 5320) with their own pandas check, opening the charts, and checking the R² values against their previous project.
    12. **Comparison with the previous version:** the Builder drafts the structural differences known from this plan:
        - one large file → modules with one job each
        - cleaning added last → cleaning done first
        - tests written after the code → tests written alongside each module
        - nothing → Docker, a Makefile and configurable paths

        `[USER]` adds anything else, such as whether the results changed.

Write each module's tests in the same step as the module, not at the end.

### Automated tests
- **Unit** (small synthetic DataFrames in `conftest.py`):
  - `config`
    - Defaults are returned when the environment variables are unset.
    - `monkeypatch` overrides are respected.
  - `load_data`
    - Loads a valid tiny CSV written to `tmp_path`.
    - Missing file → `FileNotFoundError`.
    - CSV missing a required column → `ValueError`.
  - `inspect_data`
    - Correct shape, missing counts and duplicate count on a frame with a known NaN and a known duplicate.
    - Zero duplicates case.
  - `clean_data`
    - Removes exact duplicates only, and keeps rows that differ in a single value.
    - Keeps an extreme-outlier row.
    - Does not mutate its input.
    - Idempotent (cleaning twice gives the same result).
    - Empty frame → empty frame.
  - `explore`
    - Filters return the correct rows.
    - A threshold that matches nothing returns an empty frame, not an error.
    - Group summaries have the expected index (`red`/`white`, quality levels) and values.
  - `model`
    - Synthetic data where `quality` is an exact linear function of the features → R² ≈ 1.
    - The result dict has the expected keys and `n_train + n_test == len(df)`.
    - Unknown feature → `ValueError`.
    - Too few rows → `ValueError`.
  - `plots`
    - Each function creates a non-empty PNG in `tmp_path`.
    - Creates a nested output dir that did not exist.
- **Regression** (real CSV, guards against silent behaviour changes):
  - Raw shape is 6,497 × 13, with 0 missing values and 1,177 duplicates.
  - The cleaned frame has 5,320 rows (3,961 white, 1,359 red) and no duplicates.
  - The 11-feature R² is at least the 3-feature R² (same split).
  - Both R² values fall in a loose plausible band (0.1–0.5) and are identical across two runs (determinism).
- **Integration:**
  - Set `WINE_DATA_PATH` and `WINE_OUTPUT_DIR=tmp_path` with `monkeypatch`, then call `main.main()`.
  - Assert that both PNGs exist and that the captured stdout (`capsys`) contains the inspect, cleaning, exploration and model-comparison headings.
- **Gate:** `make lint` (black `--check` plus flake8) and `make test` both pass, locally and with `make docker-test`.

### Manual Smoke Test
#### What we are proving
- A fresh setup installs cleanly.
- The analysis runs end to end from the terminal and prints readable results.
- The two charts are written to disk.
- The same analysis runs inside Docker, and its charts land in the host's `outputs/` folder through the bind mount.

#### Terminal
```bash
# 1. Setup (creates .venv with Python 3.13; the base environment is untouched)
make install
.venv/bin/python --version

# 2. Run locally, starting from an empty outputs folder
rm -f outputs/*.png
make run
ls -la outputs/

# 3. Optional: prove the env-var override works
WINE_OUTPUT_DIR=/tmp/wine_smoke make run
ls -la /tmp/wine_smoke/

# 4. Container: clear the charts, build, run, and check that they reappear on the host
#    Screenshot the docker-build output -> docs/screenshots/docker-build.png
#    Screenshot the docker-run output and the ls -> docs/screenshots/docker-run.png
rm -f outputs/*.png
make docker-build
make docker-run
ls -la outputs/
open outputs/alcohol_by_quality_boxplot.png outputs/alcohol_vs_density_scatter.png   # macOS

# 5. Optional: the same analysis via Docker Compose
rm -f outputs/*.png
docker compose up --build
ls -la outputs/
docker compose run --rm tests
docker compose down
```

#### Watch for
- `make install` creates `.venv/`, and `.venv/bin/python --version` prints `Python 3.13.x`.
- `make run` prints, in order:
  - inspection: shape `(6497, 13)`, 0 missing values, 1177 duplicates
  - cleaning: 1177 duplicates removed, `Removed 0 rows with missing values` → 5320 rows, plus the "outliers kept" note
  - filter and group summaries by type and by quality
  - a model comparison table with R² and RMSE for 3 vs 11 features (expect modest R², with 11 features ≥ 3 features)
  - the paths of the saved charts
- `ls outputs/` shows `alcohol_by_quality_boxplot.png` and `alcohol_vs_density_scatter.png`, both non-zero in size.
- Step 3 writes the charts to `/tmp/wine_smoke/` instead of `outputs/`.
- `make docker-run` prints the same numbers as the local run, and the PNGs reappear in `./outputs` after being deleted.
- Step 5:
  - `docker compose up` shows only the `analysis` service, with the same numbers as before, then exits. The PNGs reappear in `./outputs`.
  - `docker compose run --rm tests` ends with all tests passing.
- The opened charts show a boxplot that rises with quality, and a scatter with a downward-sloping trend line (more alcohol, lower density).

#### Stop
Stop and report back if any of the following happens:
- any command exits non-zero, or a Python traceback appears
- `make install` fails to find `python3.13` (rerun it as `make install PYTHON=/path/to/python3.13`)
- the printed counts differ from 6497 / 1177 / 5320
- a chart file is missing or 0 bytes
- the Docker run finishes but `./outputs` stays empty, which usually means a mount or path-quoting problem (see Risks)
- Docker cannot access the OneDrive path
- `docker compose up` also starts the `tests` service (the profile is missing), or the Compose run prints different numbers from `make docker-run` (the two configs are out of sync)

Once all checks pass, record the result in the README's "Manual smoke test result" section and save the two screenshots. Stage 1 is then complete.
