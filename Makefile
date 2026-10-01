PYTHON ?= python3.13
VENV_PY := .venv/bin/python
SRC := wine_analysis tests

.PHONY: install run test lint format clean

install:
	$(PYTHON) -m venv .venv
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.txt

run:
	$(VENV_PY) -m wine_analysis.main

test:
	$(VENV_PY) -m pytest

lint:
	$(VENV_PY) -m black --check $(SRC)
	$(VENV_PY) -m flake8 $(SRC)

format:
	$(VENV_PY) -m black $(SRC)

clean:
	rm -rf .pytest_cache
	find . -path ./.venv -prune -o -type d -name __pycache__ -exec rm -rf {} +
	rm -f outputs/*.png

.PHONY: docker-build docker-run docker-test

docker-build:
	docker build -t wine-quality .

docker-run: docker-build
	docker run --rm -v "$(CURDIR)/outputs:/app/outputs" wine-quality

docker-test: docker-build
	docker run --rm wine-quality python -m pytest -q

.PHONY: compose-up compose-test

compose-up:
	docker compose up --build && docker compose down

compose-test:
	docker compose run --rm --build tests
