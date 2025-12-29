PYTHON := uv run python
PYTEST := uv run pytest
UV_RUN := uv run
COVERAGE_THRESHOLD := 100

GREEN := \033[0;32m
RED := \033[0;31m
NC := \033[0m

.PHONY: all test lint clean help

all: test lint

test:
	@echo "$(GREEN)Running tests...$(NC)"
	@$(PYTEST) . --force-sugar --color=yes -ra \
		--cov=. --cov-report=xml:coverage.xml \
		--cov-report term-missing \
		--junitxml=pytest-junit.xml \
		--cov-fail-under=$(COVERAGE_THRESHOLD) \
		--no-cov-on-fail || (echo "$(RED)Tests failed!$(NC)"; exit 1)
	@echo "$(GREEN)Tests passed successfully!$(NC)"

retest:
	@echo "$(GREEN)Rerunning failed tests...$(NC)"
	@$(PYTEST) . --force-sugar --color=yes -ra --lf

lint:
	@echo "$(GREEN)Running linters and formatters...$(NC)"
	uv run ruff check --fix . && \
	uv run ruff format --check . && \
	uv run mypy . --strict || (echo "$(RED)Linting failed!$(NC)"; exit 1)
	@echo "$(GREEN)Linting completed successfully!$(NC)"


format:
	uv run ruff check --fix
	uv run ruff format .

clean:
	@echo "$(GREEN)Cleaning up...$(NC)"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@rm -rf .pytest_cache
	@rm -f coverage.xml pytest-junit.xml
	@echo "$(GREEN)Cleanup completed!$(NC)"
