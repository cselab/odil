RUFF_FLAGS =
BLACK_FLAGS =

default:

data:
	git clone -b data --single-branch git@github.com:cselab/odil.git data

release:
	V=$$(sed -rn 's/^version = "(.*)"$$/\1/p' pyproject.toml) && git archive --prefix="odil-$$V/" -o "odil-$$V.tar.gz" HEAD

lint:
	ruff check --fix --output-format concise $(RUFF_FLAGS) .
	black $(BLACK_FLAGS) .

.PHONY: default release lint
