# define the name of the virtual environment directory
VENV := venv
BIN := ${VENV}/bin
PYTHON := ${BIN}/python3.8
PIP := ${BIN}/pip
DIST := ${VENV}/dist
PROJECT = ${VENV}/../

.DEFAULT_GOAL := help

define PRINT_HELP_SCRIPT
import re, sys

for line in sys.stdin:
	match = re.match('^([a-zA-Z_-]+):.*?##(.*)$$', line)
	if match:
		target, help = match.groups()
		print("%20s %s", (target, help))
endef

export PRINT_HELP_SCRIPT

python_src = src scripts/*.py tests/*.py
coverage_src = src
# default target, when make executed without arguments
all: venv

$(VENV)/bin/activate: requirements.txt
	python3.8 -m venv $(VENV)
	${PIP} install --upgrade pip black isort mypy pytest coverage pylint
	${PIP} install -r requirements.txt

help:
	${PYTHON} -c "$$PRINT_HELP_SCRIPT" < "$(MAKEFILE_LIST)"

# venv is a shortcut target
venv: $(VENV)/bin/activate

check: venv check-format check-types lint

check-format:
	${BIN}/black --check ${python_src}
	${BIN}/isort --check-only ${python_src}

check-types:
	${BIN}/mypy ${python_src}

build: venv
	${PIP} install wheel
	${PYTHON} -m setup bdist_wheel

test: build
	${PIP} install ${DIST}/*
	${BIN}/coverage run --branch --source ${coverage_src} -m ${BIN}/pytest src/tests
	${BIN}/coverage report -m 

coverage: test
	${BIN}/coverage html

format: venv
	${BIN}/black ${python_src}
	${BIN}/isort ${python_src}

dev: venv
	${PIP} install -e ${PROJECT}.[dev]

lint:
	${BIN}/pylint ${python_src} -f parseable -r n

clean: clean-build clean-pyc clean-check clean-test

clean-build:
	rm -rf ${VENV}/
	rm -rf ${DIST}/
	rm -rf .eggs/
	find . -name '*.egg-info' -exec rm -rf {} +
	find . -name '*.egg' -exec rm -rf {} +

clean-pyc: # remove python artifacts
	find . -name '*.pyc' -exec rm -rf {} +
	find . -name '*.pyo' -exec rm -rf {} +
	find . -name '*~' -exec rm -rf {} +
	find . -name '__pycache__' -exec rm -rf {} +

clean-check:
	find . -name '.mypy_cache' -exec rm -rf {} +

clean-test:
	find . -name '.pytest_cache' -exec rm -rf {} +
	rm -f coverage
	rm -rf htmlcov/

.PHONY: all clean check dev build test help