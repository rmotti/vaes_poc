#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = vaes_poc
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = uv run python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync




## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	uv run ruff format --check
	uv run ruff check

## Format source code with ruff
.PHONY: format
format:
	uv run ruff check --fix
	uv run ruff format



## Run tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"




#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Download PneumoniaMNIST (64 px) into data/raw
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) -m module_vae.dataset


## Full pipeline: train -> calibrate -> evaluate -> compare (5-10 min)
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m module_vae.main


## Quick check of the pipeline (2 epochs, 1 seed, no baselines)
.PHONY: quick
quick:
	$(PYTHON_INTERPRETER) -m module_vae.main --epochs 2 --n-seeds 1 --no-baselines


## Score test exams with the saved model
.PHONY: predict
predict:
	$(PYTHON_INTERPRETER) -m module_vae.inference


## Latent size and beta experiments (needs `make train` first)
.PHONY: ablation
ablation:
	$(PYTHON_INTERPRETER) -m module_vae.modeling.ablation


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
