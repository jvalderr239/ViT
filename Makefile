.ONESHELL:
# define the name of the virtual environment directory
VENV := venv
BIN := ${VENV}/bin
PYTHON := ${BIN}/python3.8 -m
DIST := ${VENV}/dist
PIP := ${PYTHON} pip install --upgrade
LIBS := ${VENV}/lib/python3.8/site-packages
VENV_PROJECT := ${VENV}/vit
PROJECT := ${VENV}/../
BUILD := ./build

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

$(VENV)/bin/activate: requirements.txt setup.py
	test -d $(VENV) || python3.8 -m venv $(VENV)
	${PIP} pip black isort mypy pytest coverage pylint 
	${PIP} -r requirements.txt

help:
	${PYTHON} -c "$$PRINT_HELP_SCRIPT" < "$(MAKEFILE_LIST)"

# venv is a shortcut target
venv: $(VENV)/bin/activate

check: venv check-format check-types lint

check-format:
	${PYTHON} black --check ${python_src}
	${PYTHON} isort --check-only ${python_src} 

check-types:
	${PYTHON} mypy ${python_src} 

build: venv
	${PIP} wheel 
	${PYTHON} setup bdist_wheel -d ${DIST} 

install: build
	test -d ${VENV_PROJECT} || ${PIP} ${DIST}/* --no-deps --target ${VENV_PROJECT} 
	test -d ${VENV_PROJECT}/packages || ${PIP} -r requirements.txt --target ${VENV_PROJECT}/packages

test: clean install
	${PYTHON} coverage run --branch --source ${coverage_src} -m pytest ${VENV_PROJECT}/tests 
	${PYTHON} coverage report 

coverage: test
	${PYTHON} coverage html

format: venv
	${PYTHON} black ${python_src}
	${PYTHON} isort ${python_src}

dev: 
	pip install -e ${PROJECT}.[dev] 

lint:
	${PYTHON} pylint ${python_src} -f parseable -r n

clean: clean-build clean-pyc clean-check clean-test

clean-build:
	rm -rf ${VENV}/
	rm -rf ${DIST}/
	rm -rf build/
	rm -rf out/
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
	rm -f .coverage
	rm -rf htmlcov/

.PHONY: all clean check dev build test help