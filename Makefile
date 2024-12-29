.PHONY: format format_check static test test_coverage security
all: format static test test_coverage security

format:
	black --line-length 100 .
	isort --multi-line 3 --trailing-comma --force-grid-wrap 0 --use-parentheses --line-width 100 .
format_check:
	black --line-length 100 --check .
	isort --multi-line 3 --trailing-comma --force-grid-wrap 0 --use-parentheses --line-width 100 . --check-only
static:
	flake8 --ignore E501,E203,W503 --max-line-length=100
	mypy --ignore-missing-imports --no-strict-optional --no-site-packages .
test:
	python -m unittest -v
test_coverage:
	coverage run -m unittest && coverage report -m -i
security:
	bandit -r