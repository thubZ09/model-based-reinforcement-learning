.PHONY: help install lint format test test-all train-dreamer train-muzero train-mbpo train-pets train-tdmpc train-jepa plot viz clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies (uv)
	uv sync --all-extras

install-pip: ## Install dependencies (pip)
	pip install -e ".[dev]"

lint: ## Run ruff linter
	ruff check .

format: ## Run ruff formatter
	ruff format .

format-check: ## Check formatting without changes
	ruff format --check .

test: ## Run tests
	python -m pytest tests/ -v

test-all: ## Run all tests with coverage
	python -m pytest tests/ -v --cov=. --cov-report=term-missing

train-muzero: ## Run MuZero on CartPole
	python muzero_cartpole.py

train-muzero-ez: ## Run MuZero + EfficientZero on CartPole
	python muzero_efficientzero_cartpole.py

train-mbpo-cheetah: ## Run MBPO on HalfCheetah
	python mbpo_halfcheetah.py

train-mbpo-hopper: ## Run MBPO on Hopper
	python mbpo_hopper.py

train-pets-cartpole: ## Run PETS on CartPole
	python pets_cartpole.py

train-pets-pendulum: ## Run PETS on Pendulum
	python pets_pendulum.py

train-tdmpc2: ## Run TD-MPC2 on Pendulum
	python tdmpc2_pendulum.py

train-dreamer: ## Run DreamerV3 on DMControl walker_walk
	python -m dreamerv3.train

train-jepa: ## Run JEPA on N-body physics
	python jepa_nbody/train.py

plot: ## Generate plots from log files
	python plot_results.py

viz: ## Watch trained agent
	python viz.py

record: ## Record trained agent as GIF/MP4
	python record_agent.py

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf build/ dist/ *.egg-info .eggs
	rm -rf results/ logs/ outputs/
