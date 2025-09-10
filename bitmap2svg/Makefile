.PHONY: setup test fmt bench serve watch
setup:
	uv venv -p python3.11 || true
	pip install -U pip
	pip install -e .

test:
	pytest -q

fmt:
	ruff check --fix .
	python -m pip install -q pyink && pyink -m .

bench:
	python -m bitmap2svg.cli.bench tests/assets

serve:
	python -m bitmap2svg.service

watch:
	python -m bitmap2svg.workers.watcher in/ out/