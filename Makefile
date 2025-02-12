.PHONY: clean create-env activate-venv refresh-packages save-dependencies format lint check commit push

GREEN=\033[0;32m
BLUE=\033[0;34m
YELLOW=\033[1;33m
NC=\033[0m # No Color


clean::
	@rm -f -r .venv

create-env::
	python -m venv venv

activate-venv::
	venv/Scripts/Activate.ps1

refresh-packages::
	pip install -r requirements.txt

save-dependencies::
	pip freeze > requirements.txt

format::
	ruff format .
	ruff check . --fix

lint::
	ruff check .
	ruff format . --check

check:: format lint
	@echo "\n${YELLOW}Formatting and linting complete${NC}\n"

commit::
	git add .
	cz commit

push::
	(git pull || true) && git push

