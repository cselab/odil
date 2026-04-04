RUFF_FLAGS =
FORMAT_FLAGS =
MYPY_FLAGS =

MYPY_TARGETS = \
    src/odil/history.py \

default: lint format

data:
	git clone -b data --single-branch git@github.com:cselab/odil.git data

release:
	V=$$(sed -rn 's/^version = "(.*)"$$/\1/p' pyproject.toml) && git archive --prefix="odil-$$V/" -o "odil-$$V.tar.gz" HEAD

lint:
	ruff check --fix $(RUFF_FLAGS)

format:
	ruff format $(FORMAT_FLAGS)

check: lint format

mypy:
	mypy $(MYPY_FLAGS) $(MYPY_TARGETS)

.PHONY: default data release lint format check mypy
