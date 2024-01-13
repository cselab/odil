default:

data:
	git clone -b data --single-branch git@github.com:cselab/odil.git data

release:
	V=$$(sed -rn 's/^version = "(.*)"$$/\1/p' pyproject.toml) && git archive --prefix="odil-v$$V/" -o "odil-v$$V.tar.gz" HEAD

.PHONY: default release
