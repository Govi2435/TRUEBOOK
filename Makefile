PYTHON := python3
PIP := $(PYTHON) -m pip
VENV_DIR := .venv
ACTIVATE := . $(VENV_DIR)/bin/activate

create-venv:
	@test -d $(VENV_DIR) || $(PYTHON) -m venv $(VENV_DIR)
	@echo "[ok] venv ready"

install-core: create-venv
	@$(ACTIVATE) && $(PIP) install --upgrade pip
	@$(ACTIVATE) && $(PIP) install -r requirements-core.txt
	@echo "[ok] core deps installed"

install-ml: create-venv
	@$(ACTIVATE) && $(PIP) install -r requirements-ml.txt
	@echo "[ok] ml deps installed"

run-api:
	@$(ACTIVATE) && uvicorn services.api.main:app --host 0.0.0.0 --port 8000 --reload

build-index:
	@$(ACTIVATE) && $(PYTHON) scripts/build_index.py

ingest-sample:
	@$(ACTIVATE) && $(PYTHON) scripts/ingest_sample.py

lint:
	@echo "(lint placeholder)"

test:
	@$(ACTIVATE) && pytest -q