
SHELL=/bin/bash

all: lint_node lint_python

TARGET_DIRS:=./whispering

flake8:
	find $(TARGET_DIRS) | grep '\.py$$' | xargs flake8
black:
	find $(TARGET_DIRS) | grep '\.py$$' | xargs black --diff | python ./scripts/check_null.py
isort:
	find $(TARGET_DIRS) | grep '\.py$$' | xargs isort --diff | python ./scripts/check_null.py
pydocstyle:
	find $(TARGET_DIRS) | grep -v tests | xargs pydocstyle --ignore=D100,D101,D102,D103,D104,D105,D107,D203,D212
pytest:
	pytest
	
yamllint:
	find . \( -name node_modules -o -name .venv \) -prune -o -type f -name '*.yml' -print \
		| xargs yamllint --no-warnings

version_check:
	 git tag | python ./scripts/check_version.py --toml pyproject.toml -i README.md --tags -

lint_python: flake8 black isort pydocstyle version_check pytest


pyright:
	npx pyright

markdownlint:
	find . -type d \( -name node_modules -o -name .venv \) -prune -o -type f -name '*.md' -print \
	| xargs npx markdownlint --config ./.markdownlint.json

lint_node: markdownlint pyright


style:
	find $(TARGET_DIRS) | grep '\.py$$' | xargs black
	find $(TARGET_DIRS) | grep '\.py$$' | xargs isort
