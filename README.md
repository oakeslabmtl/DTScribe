# DTScribe: Generating Models for Digital Twin Reporting using Local Large Language Models

## Installation

### 1. Install Ollama
Follow the [official instructions](https://ollama.ai/download) for your operating system.  
Once installed, pull the required models:

```bash
# Example LLM (adjust to desired model, e.g., llama3, qwen, mistral, etc.):
ollama pull llama3.2
# Embedding models:
ollama pull nomic-embed-text
ollama pull embeddinggemma
```

### 2. (Optional) Install uv
[uv](https://docs.astral.sh/uv/) is used as the Python package/dependency manager.

Linux/macOS:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Windows PowerShell:
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Install project dependencies
```bash
uv sync
```
or
```bash
pip install -e .
```


## Running

To run the extraction, you need to do:
```bash
uv run src\experiment_runner.py
```
