.PHONY: proto test lint clean install

proto:
	python -m grpc_tools.protoc \
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
	python -c "import shutil, glob, os; [shutil.rmtree(p, ignore_errors=True) for p in ['build', 'dist', '.pytest_cache', '__pycache__'] + glob.glob('*.egg-info')]; [os.remove(p) for p in glob.glob('**/*.pyc', recursive=True)]"
