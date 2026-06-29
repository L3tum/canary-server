.PHONY: install run test clean help

# Default variables
HOST ?= 0.0.0.0
PORT ?= 8000
API_KEY ?= your-api-key-here
PARALLEL_SIZE ?= 1
THREAD_POOL_SIZE ?=

# Help target
help:
	@echo "Available targets:"
	@echo "  install     - Install dependencies"
	@echo "  run         - Run the ASR server"
	@echo "  run-cpu-opt - Run the ASR server with CPU optimizations"
	@echo "  test        - Test the ASR endpoint"
	@echo "  clean       - Clean Python cache files"
	@echo "  help        - Show this help message"

# Install dependencies
install:
	uv pip install -r requirements.txt

# Run the server
run:
	INTERNAL_API_KEY=$(API_KEY) python -m src.nemo_openai_server --host $(HOST) --port $(PORT) --parallel-size $(PARALLEL_SIZE)

# Run the server with CPU optimizations
run-cpu-opt:
	INTERNAL_API_KEY=$(API_KEY) python -m src.nemo_openai_server --host $(HOST) --port $(PORT) --parallel-size $(PARALLEL_SIZE) --thread-pool-size $(THREAD_POOL_SIZE)

# Test the endpoint (legacy)
test-endpoint:
	python tests/test_endpoint.py --api-key $(API_KEY) --host $(HOST) --port $(PORT)

# Run all pytest tests
test:
	python -m pytest tests -v --tb=short

# Clean Python cache files
clean:
	find . -type f -name "*.py[cod]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .eggs/

# Install in development mode
develop:
	uv pip install -e .

# Lint code with ruff
lint:
	python -m ruff check src/ tests/

# Auto-format with ruff
format:
	python -m ruff format src/ tests/

# Start development server with hot reload
dev:
	INTERNAL_API_KEY=$(API_KEY) MODEL_NAME=$(MODEL_NAME) python -m uvicorn src.nemo_openai_server:app --host $(HOST) --port $(PORT) --reload

# Build Docker image
docker-build:
	docker build -t canary-asr-server .