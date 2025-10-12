RUFF_FLAGS =
BLACK_FLAGS =
MYPY_FLAGS =

MYPY_TARGETS = \
    src/odil/history.py \

default:

data:
	git clone -b data --single-branch git@github.com:cselab/odil.git data

release:
	V=$$(sed -rn 's/^version = "(.*)"$$/\1/p' pyproject.toml) && git archive --prefix="odil-$$V/" -o "odil-$$V.tar.gz" HEAD

lint:
	ruff check --fix $(RUFF_FLAGS) .
	black $(BLACK_FLAGS) .

mypy:
	mypy $(MYPY_FLAGS) $(MYPY_TARGETS)

.PHONY: default release lint mypy
