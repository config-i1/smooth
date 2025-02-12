.PHONY: all lint test install environment

SRC_DIR = smooth

lint: 
	flake8 $(SRC_DIR)
	pydocstyle $(SRC_DIR)

test:
	pytest $(SRC_DIR)

install:
	pip install -e ".[dev]"

environment:
	(\
		echo "> Creating venv"; \
		python -m venv .venv; \
		source .venv/bin/activate; \
		echo "> Installing local package in editable mode"; \
		pip install -e ".[dev]"; \
		echo "> Making venv available in jupyter notebooks"; \
		python -m ipykernel install --name=$(SRC_DIR); \
		jupyter kernelspec list; \
		echo "> Installing pre-commit"; \
		pre-commit install; \
	)

clean:
	echo "> Removing virtual environment"
	rm -r .venv
	echo "> Uninstalling from jupyter"
	jupyter kernelspec uninstall $(SRC_DIR)
