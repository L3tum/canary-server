.PHONY: install run test clean help

# Default variables
HOST ?= 0.0.0.0
PORT ?= 8000
API_KEY ?= your-api-key-here

# Help target
help:
	@echo "Available targets:"
	@echo "  install     - Install dependencies"
	@echo "  run         - Run the ASR server"
	@echo "  test        - Test the ASR endpoint"
	@echo "  clean       - Clean Python cache files"
	@echo "  help        - Show this help message"

# Install dependencies
install:
	pip install -r requirements.txt

# Run the server
run:
	INTERNAL_API_KEY=$(API_KEY) python -m src.nemo_openai_server --host $(HOST) --port $(PORT)

# Test the endpoint
test:
	python tests/test_endpoint.py --api-key $(API_KEY) --host $(HOST) --port $(PORT)

# Clean Python cache files
clean:
	find . -type f -name "*.py[cod]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .eggs/

# Install in development mode
develop:
	pip install -e .