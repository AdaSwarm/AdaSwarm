install:
	poetry install
test:
	poetry run python -m unittest tests/unit/test_* 
lint:
	poetry run pylint -d duplicate-code adaswarm/**/*.py
run: install
	poetry run examples/main.py
build:
	poetry build
publish: build
	poetry publish
	