# -------- config knobs --------
PY ?= python
OUT_DIR ?= data/out
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
.PHONY: help data-dryrun data clean checksums data-course-dryrun data-course

help:
	@echo "Targets:"
	@echo "  make data-dryrun   # Preview STAC results (no files written)"
	@echo "  make data          # Build scenes.parquet + scene splits + CHECKSUMS"
	@echo "  make data-course-dryrun   # Preview STAC results (no files written)"
	@echo "  make data-course          # Build scenes.parquet + scene splits + CHECKSUMS"
	@echo "  make clean         # Remove data/out"
	@echo ""
	@echo "Vars you can override:"
	@echo "  AOI=<path.geojson> START=YYYY-MM-DD END=YYYY-MM-DD CLOUD=25 MAX_SCENES=30 SIGN=--sign-assets"

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

checksums:
	@test -f $(OUT_DIR)/CHECKSUMS.md && cat $(OUT_DIR)/CHECKSUMS.md || echo "No CHECKSUMS.md yet."