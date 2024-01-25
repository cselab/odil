PYTHON = python3
BACKEND = tf
DTYPE = float64
MAKE = make --no-print-directory
ARGS =

default: all

all: test_domain test_mg_interp test_mg_restrict test_optimize test_optimize_32 test_newton test_io

test_domain:
test_mg_interp:
test_mg_restrict:
test_optimize:
test_optimize_32:
	@$(MAKE) test_optimize DTYPE=float32
test_optimize_lbfgs:
	@$(MAKE) test_optimize ARGS='--optimizers lbfgs'
test_newton:
test_newton_jax:
	@echo "Skip $@, not implemented"

test_io:
	$(PYTHON) -m unittest ./test_io.py

test_%: test_%.py
	ODIL_BACKEND=$(BACKEND) ODIL_DTYPE=$(DTYPE) $(PYTHON) $< $(ARGS)

%_tf:
	@$(MAKE) $* BACKEND=tf

%_jax:
	@$(MAKE) $* BACKEND=jax

.PHONY: default all
