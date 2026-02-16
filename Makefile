.PHONY: proto test lint clean install

proto:
	python3 -m grpc_tools.protoc \
		-I proto \
		--python_out=src/avp \
		proto/avp.proto

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache __pycache__
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '*.pyc' -delete
