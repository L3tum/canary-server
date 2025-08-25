# Using uv with NeMo OpenAI ASR Server

This project now supports [uv](https://docs.astral.sh/uv/), a fast Python package installer and resolver, as an alternative to conda for managing dependencies.

## Why uv?

uv is extremely fast compared to traditional Python package managers:
- Up to 10-100x faster than pip
- Up to 200x faster than conda
- Single binary installation
- Compatible with the existing Python ecosystem

## Installing uv

Follow the official installation instructions at https://docs.astral.sh/uv/getting-started/installation/

On Linux and macOS, you can typically install it with:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setting up the environment with uv

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd nemo_openai_server
   ```

2. Create a virtual environment:
   ```bash
   uv venv
   ```
   
3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

## Running the server with uv

After setting up the environment, you can run the server as usual:
```bash
python -m src.nemo_openai_server --api-key your-secret-api-key
```

## Using the provided scripts

The project includes setup scripts that have been updated to use uv:
- `setup_asr_endpoint.sh` - Sets up the environment and dependencies using uv
- `run_asr_endpoint.sh` - Runs the server (activate the virtual environment first)

## Development workflow

For development, you can install the package in editable mode:
```bash
uv pip install -e .
```

Or with development dependencies:
```bash
uv pip install -e ".[dev]"
```

## Docker builds

The Dockerfile has been updated to use uv for faster dependency installation in container builds.