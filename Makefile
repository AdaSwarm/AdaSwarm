test:
	poetry run python -m unittest tests/unit/test_* 
lint:
	poetry run pylint -d duplicate-code examples/main.py adaswarm/particle.py adaswarm/data.py \
	adaswarm/model.py adaswarm/utils/matrix.py adaswarm/utils/options.py adaswarm/utils/progressbar.py \
	adaswarm/rempso.py
