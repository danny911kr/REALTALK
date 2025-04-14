SHELL=/usr/bin/env bash

# Path variables
PROJECT_ROOT=$(shell pwd)
SRC_ROOT=$(PROJECT_ROOT)
MAKEFILE_PATH:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
TEST_DIR= $(MAKEFILE_PATH)/tests

# Specify the names of all executables to make.
PROG=init precommit config fix check format update install
.PHONY: ${PROG}

ifneq (,$(wildcard .env))
    include .env
    export $(shell grep -v '^\#' .env | xargs)
endif

init: install config

precommit: format fix check

config:
	git config commit.template ~/.gitmessage

fix:
	ruff check --fix $(SRC_ROOT)

check:
	ruff check $(SRC_ROOT)
	pyright $(SRC_ROOT)
	yamllint $(SRC_ROOT)

format:
	ruff format $(SRC_ROOT)
	yamlfix $(SRC_ROOT)

update:
	pip install --upgrade pip
	pip install --upgrade -r requirements.txt -r requirements-dev.txt

install:
	pip install --upgrade pip
	pip install -r requirements.txt -r requirements-dev.txt