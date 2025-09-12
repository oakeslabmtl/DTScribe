# LLM to Digital Twin Description Framework (LLM2DTDF)

## Installation

- Install Ollama
    - Download LLM models
- Install `uv`
- TODO: complete

## Running

To run the extraction, you need to do:
```bash
uv run src\main.py
```

Here is the full list of available parameters:
```python
parser.add_argument("--mode", choices=["both", "extraction", "oml"], default="both", help="Run extraction, OML generation, or both")
parser.add_argument("--pdf", default="data/papers/The Incubator Case Study for Digital Twin Engineering.pdf", help="PDF path for extraction")
parser.add_argument("--chunk-size", type=int, default=1500)
parser.add_argument("--chunk-overlap", type=int, default=200)
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument("--model-name", default="qwen3:4b")
parser.add_argument("--embedding-model", default="nomic-embed-text")
parser.add_argument("--source-experiment-id", help="Existing experiment id (hash_timestamp or just hash for latest) containing characteristics for standalone OML generation")
parser.add_argument("--no-save", action="store_true", help="Do not persist results")
```

## Analyzing results

Full command for analyzing results and generating a report/figures.
```bash
uv run src\results_visualizer.py analyze --experiments-dir experiments --dashboard --show --report
```