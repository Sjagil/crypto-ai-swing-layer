.PHONY: install dev doctor smoke test lint refs

install:
	python -m pip install -r requirements.txt
	python -m pip install -e .

dev:
	python -m pip install -r requirements.txt
	python -m pip install -r requirements/dev.txt
	python -m pip install -e .

doctor:
	python -m crypto_ai_swing.cli doctor

smoke:
	python -m crypto_ai_swing.cli smoke

test:
	python -m pytest -q

lint:
	python -m ruff check src tests scripts

refs:
	python scripts/clone_references.py
