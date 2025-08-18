# -------- config knobs --------
PY ?= python

# Conda environment settings
ENV_NAME ?= geoAI
CONDA_RUN ?= conda run -n $(ENV_NAME)
CONDA_RUN_PY ?= $(CONDA_RUN) python

# Minimum required Python version (must match pyproject.toml [project].requires-python)
PY_MIN_MAJOR ?= 3
PY_MIN_MINOR ?= 10
OUT_DIR ?= data/out
GEO_BENCH_DIR ?= data/geobench
STAC_URL ?= https://planetarycomputer.microsoft.com/api/stac/v1
COLLS ?= sentinel-2-l2a
AOI ?= data/aois/central_coast_inland.geojson
START ?= 2020-06-01
END ?= 2020-09-30
CLOUD ?= 25
MAX_SCENES ?= 30
STRATIFY ?= month aoi
SIGN ?=            # set to "--sign-assets" to enable MPC signing

# ===== Course presets =====
COURSE_AOI       ?= data/aois/central_coast_inland.geojson
COURSE_START     ?= 2020-01-01
COURSE_END       ?= 2020-12-31
COURSE_CLOUD     ?= 40
COURSE_MAX       ?= 200
COURSE_COLLS     ?= sentinel-2-l2a
COURSE_STRATIFY  ?= month aoi
COURSE_SIGN      ?=     # set to "--sign-assets" if needed
COURSE_TARGET    ?= 0     # set to e.g. 120 to cap total scenes globally

# -------- convenience bundles --------
STAC_ARGS = --stac-url $(STAC_URL) --collections $(COLLS) \
            --aoi-file $(AOI) --start $(START) --end $(END) \
            --cloud-cover-max $(CLOUD) --max-scenes-per-aoi $(MAX_SCENES) \
            --stratify $(STRATIFY)

# -------- targets --------
.PHONY: help env ensure-env check-python install install-dev kernelspec test test-shortcode docs docs-full docs-one preview clean data-dryrun data checksums data-course-dryrun data-course geogfm-clean geogfm-clean-all geobench

help:
	@echo "Targets:"
	@echo "  make env           # Reminder to activate conda env (geoAI)"
	@echo "  make install       # pip install -e . (editable geogfm)"
	@echo "  make install-dev   # editable install + pytest"
	@echo "  make kernelspec    # register Jupyter kernel 'geoai' for the env"
	@echo "  make test          # run pytest"
	@echo "  make docs          # incremental docs build"
	@echo "  make docs-full     # full docs build (clears cache)"
	@echo "  make docs-one FILE=path.qmd  # render only the specified file (or FILES=\"a.qmd b.qmd\")"
	@echo "  make bootstrap-lib # render foundational pages in order"
	@echo "  make preview       # quarto preview (in book/)"
	@echo "  make clean         # remove data/out or clean docs intermediates"
	@echo "  make geobench      # create data/geobench, download GEO-Bench there, and print setup instructions"
	@echo "  make geogfm-clean   # delete generated .py under geogfm/ (keeps __init__.py)"
	@echo "  make geogfm-clean-all # delete ALL .py under geogfm/ (dangerous)"
	@echo "  (Tangling is now handled during Quarto render via the Python panflute filter)"
	@echo ""
	@echo "Dataset targets:"
	@echo "  make data-dryrun         # Preview STAC results (no files written)"
	@echo "  make data                # Build scenes.parquet + scene splits + CHECKSUMS"
	@echo "  make data-course-dryrun  # Preview STAC results (no files written)"
	@echo "  make data-course         # Build scenes.parquet + scene splits + CHECKSUMS"
	@echo ""
	@echo "Vars you can override:"
	@echo "  AOI=<path.geojson> START=YYYY-MM-DD END=YYYY-MM-DD CLOUD=25 MAX_SCENES=30 SIGN=--sign-assets"

env:
	@echo "Use: conda activate geoAI"

# Ensure the conda environment exists (idempotent)
ensure-env:
	@conda env list | grep -q "$(ENV_NAME)" \
		&& echo "Conda env '$(ENV_NAME)' exists" \
		|| (echo "Creating conda env '$(ENV_NAME)' from environment.yml..." && conda env create -f environment.yml -n $(ENV_NAME))

# Guard: ensure the selected Python interpreter meets the minimum version
check-python:
	@$(PY) -c "import sys; req=(int('${PY_MIN_MAJOR}'),int('${PY_MIN_MINOR}')); cur=sys.version_info; \
	import sys as s; \
	(print(f'Python version OK: {cur.major}.{cur.minor}.{cur.micro} ({sys.executable})') if cur>=req else ( \
	print(f'Error: Python {req[0]}.{req[1]}+ is required, found {cur.major}.{cur.minor}.{cur.micro} at: {sys.executable}'), \
	print('Hint: Use make install-dev to create/update the geoAI conda env, or activate it with conda activate geoAI.'), \
	s.exit(1)))"

install: ensure-env
	@$(MAKE) check-python PY="$(CONDA_RUN_PY)"
	$(CONDA_RUN_PY) -m pip install -e .

install-dev: ensure-env
	@$(MAKE) check-python PY="$(CONDA_RUN_PY)"
	$(CONDA_RUN_PY) -m pip install -e .
	$(CONDA_RUN_PY) -m pip install -U pytest

kernelspec: ensure-env
	$(CONDA_RUN_PY) -m ipykernel install --user --name geoai --display-name "geoai"

test:
	@echo "Running pytest $(if $(TESTS),for '$(TESTS)',for all tests) with verbose output..."
	@set -e; \
	pytest -vv -ra $(TESTS) || rc=$$?; \
	if [ -n "$$rc" ]; then \
	  if [ $$rc -eq 5 ]; then \
	    echo "Pytest exit code 5 (no tests collected or all skipped) — treating as success"; \
	    exit 0; \
	  else \
	    exit $$rc; \
	  fi; \
	fi

test-shortcode:  ## Test path shortcode filter functionality
	@echo "🧪 Testing path shortcode filter..."
	@pytest -xvs tests/test_path_shortcode_filter.py

docs:
	cd book && $(PY) build_docs.py

docs-full:
	cd book && $(PY) build_docs.py --full

docs-one:
	@if [ -z "$(FILE)$(FILES)" ]; then \
	  echo "Usage: make docs-one FILE=chapters/your_file.qmd"; \
	  echo "   or: make docs-one FILES=\"chapters/a.qmd chapters/b.qmd\""; \
	  exit 2; \
	fi
	cd book && $(PY) build_docs.py --only $(if $(FILES),$(FILES),$(FILE))

bootstrap-lib:
	cd book && $(PY) build_docs.py --bootstrap

preview:
	cd book && quarto preview

data-dryrun:
	$(PY) data/build_from_stac.py $(STAC_ARGS) $(SIGN) --out-dir $(OUT_DIR) --dryrun


data-course-dryrun:
	$(PY) data/build_from_stac.py \
	  --stac-url $(STAC_URL) \
	  --collections $(COURSE_COLLS) \
	  --aoi-file $(COURSE_AOI) \
	  --start $(COURSE_START) --end $(COURSE_END) \
	  --cloud-cover-max $(COURSE_CLOUD) \
	  --max-scenes-per-aoi $(COURSE_MAX) \
	  --stratify $(COURSE_STRATIFY) \
	  $(COURSE_SIGN) \
	  $(if $(filter-out 0,$(COURSE_TARGET)),--target-total-scenes $(COURSE_TARGET)) \
	  --out-dir $(OUT_DIR) \
	  --dryrun

data-course:
	@mkdir -p $(OUT_DIR)
	$(PY) data/build_from_stac.py \
	  --stac-url $(STAC_URL) \
	  --collections $(COURSE_COLLS) \
	  --aoi-file $(COURSE_AOI) \
	  --start $(COURSE_START) --end $(COURSE_END) \
	  --cloud-cover-max $(COURSE_CLOUD) \
	  --max-scenes-per-aoi $(COURSE_MAX) \
	  --stratify $(COURSE_STRATIFY) \
	  $(COURSE_SIGN) \
	  $(if $(filter-out 0,$(COURSE_TARGET)),--target-total-scenes $(COURSE_TARGET)) \
	  --out-dir $(OUT_DIR)

data:
	@mkdir -p $(OUT_DIR)
	$(PY) data/build_from_stac.py $(STAC_ARGS) $(SIGN) --out-dir $(OUT_DIR)

clean:
	rm -rf $(OUT_DIR)
	cd book && $(PY) build_docs.py --clean

checksums:
	@test -f $(OUT_DIR)/CHECKSUMS.md && cat $(OUT_DIR)/CHECKSUMS.md || echo "No CHECKSUMS.md yet."

# ===== Literate module export =====
GEOGFM_DIR ?= geogfm

geogfm-clean:
	@echo "Deleting generated Python files under $(GEOGFM_DIR) (excluding __init__.py)..."
	@find $(GEOGFM_DIR) -type f -name "*.py" ! -name "__init__.py" -print -delete || true

geogfm-clean-all:
	@echo "DELETING ALL Python files under $(GEOGFM_DIR) (including __init__.py)..."
	@find $(GEOGFM_DIR) -type f -name "*.py" -print -delete || true

# ===== GEO-Bench data setup =====
geobench:
	@mkdir -p $(GEO_BENCH_DIR)
	@echo "📦 Downloading GEO-Bench into $(GEO_BENCH_DIR) ..."
	@GEO_BENCH_DIR="$(PWD)/$(GEO_BENCH_DIR)" $(CONDA_RUN) geobench-download | cat
	@echo ""
	@echo "To use GEO-Bench in notebooks, set the environment variable in-session:"
	@echo "  >>> import os; os.environ['GEO_BENCH_DIR'] = '$(PWD)/$(GEO_BENCH_DIR)'"
	@echo "Or in your shell before launching Jupyter:"
	@echo "  export GEO_BENCH_DIR='$(PWD)/$(GEO_BENCH_DIR)'"
	@echo ""
	@echo "Done. If you encounter permission or network issues, re-run: make geobench"
