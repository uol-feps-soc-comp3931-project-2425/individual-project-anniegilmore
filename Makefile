.PHONY: clean create-env refresh-packages refresh-template reconfigure-template generate-standard-template format lint check validate-tests test test-no-output-file test-one commit push
-include *Additional*.mk
-include *Pre*.mk

GREEN=\033[0;32m
BLUE=\033[0;34m
YELLOW=\033[1;33m
NC=\033[0m # No Color


clean::
	@rm -f -r .venv
	@rm -f -r std_template
	@rm -f -r .pytest_cache
	@rm -f -r .mypy_cache
	@rm -f -r .ruff_cache
	@rm -f -r htmlcov
	@rm -f -r logs
	@rm -f .coverage
	@rm -f .python-version
	@rm -f poetry.lock

setup-githooks::
	git config core.hooksPath .githooks
	chmod +x .githooks/*

create-env::
	pyenv local 3.11
	poetry install --all-extras --no-root

refresh-packages::
	poetry up --latest

refresh-template:: setup-githooks
	cruft update -r

reconfigure-template::
	cruft update -r -i

generate-standard-template::
	PYTHONPATH=. PYTHONDEVMODE=1 python tools/generate_standard_template.py

format::
	ruff format .
	ruff check . --fix

lint::
	ruff check .
	ruff format . --check
	MYPYPATH=src/ mypy tools
	MYPYPATH=src/ mypy src
	MYPYPATH=src/ mypy tests
	MYPYPATH=src/ mypy spikes

check:: format lint
	@echo "\n${YELLOW}Formatting and linting complete${NC}\n"

validate-tests::
	PYTHONPATH=./src:. PYTHONDEVMODE=1 pytest --collect-only tests

test::
	mkdir -p logs/tests
	PYTHONPATH=./src:. PYTHONDEVMODE=1 pytest -m "not slow" tests | tee logs/tests/$(shell date +'%Y_%m_%d_%H_%M_%S').log

test-all::
	mkdir -p logs/tests
	PYTHONPATH=./src:. PYTHONDEVMODE=1 pytest tests | tee logs/tests/$(shell date +'%Y_%m_%d_%H_%M_%S').log

test-one::
	mkdir -p logs/tests
	PYTHONPATH=./src:. PYTHONDEVMODE=1 pytest tests -k $(TEST) | tee logs/tests/$(shell date +'%Y_%m_%d_%H_%M_%S').log

test-find-n-slowest::
	PYTHONPATH=./src:. PYTHONDEVMODE=1 pytest --durations=${N} tests

commit::
	.githooks/pre-commit
	git add .
	cz commit

push::
	(git pull || true) && git push

-include *Post*.mk
