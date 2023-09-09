PYTHON = python3
BACKEND = tf
DTYPE = float64
MAKE = make --no-print-directory
ARGS =

default: all

all: test_domain test_interp test_restrict test_optimize test_optimize_32

test_domain:
test_interp:
test_restrict:
test_optimize:
test_optimize_32:
	@$(MAKE) test_optimize DTYPE=float32
test_optimize_lbfgs:
	@$(MAKE) test_optimize ARGS='--optimizers lbfgs'

test_%: test_%.py
	ODIL_BACKEND=$(BACKEND) ODIL_DTYPE=$(DTYPE) $(PYTHON) $< $(ARGS)

%_tf:
	@$(MAKE) $* BACKEND=tf

%_jax:
	@$(MAKE) $* BACKEND=jax

.PHONY: default all
