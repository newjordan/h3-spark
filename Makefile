PYTHON ?= python3

.PHONY: test list-native build-native

test:
	$(PYTHON) -m pytest

list-native:
	bash scripts/build_native.sh --list

build-native:
	bash scripts/build_native.sh
