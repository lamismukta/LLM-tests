# LLM CV Analysis Pipeline

A modular framework for comparing different LLM models and analysis strategies when evaluating CVs.

## Features

- **Multiple Pipeline Strategies**:
  - **One-Shot**: Single comprehensive prompt analysis
  - **Chain-of-Thought**: Step-by-step reasoning approach
  - **Multi-Layer**: Iterative refinement with multiple analysis stages

- **Easy LLM Switching**: Abstract provider interface makes it simple to switch between LLM providers (currently OpenAI, easily extensible)

- **Comparison Framework**: Built-in tools to compare results across pipelines and models

## Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set up environment variables**:
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```

3. **Configure settings** (optional):
Edit `config.yaml` to customize models, pipelines, and analysis settings.

## Usage

### Basic Usage

Run all pipelines on all CVs with all configured models:
```bash
python run_analysis.py
```

### Quick Test

Run a quick test on the first 3 CVs:
```bash
python run_analysis.py --quick-test
```

### Custom Experiments

Run specific models:
```bash
python run_analysis.py --models gpt-4o-mini gpt-4o
```

Run specific pipelines:
```bash
python run_analysis.py --pipelines one_shot chain_of_thought
```

Analyze specific CVs:
```bash
python run_analysis.py --cv-ids A1 A2 B1
```

Combine options:
```bash
python run_analysis.py --models gpt-4o-mini --pipelines one_shot --cv-ids A1 A2 --experiment-name my_test
```

## Project Structure

```
.
├── src/
│   ├── providers/          # LLM provider implementations
│   │   ├── base.py         # Abstract base class
│   │   └── openai_provider.py
│   ├── pipelines/          # Analysis pipeline strategies
│   │   ├── base.py
│   │   ├── one_shot.py
│   │   ├── chain_of_thought.py
│   │   └── multi_layer.py
│   └── comparison.py       # Comparison and evaluation framework
├── data/                   # CV data files
├── results/                # Experiment results (generated)
├── config.yaml            # Configuration file
├── requirements.txt        # Python dependencies
└── run_analysis.py        # Main execution script
```

## Adding New LLM Providers

1. Create a new provider class in `src/providers/` that inherits from `LLMProvider`
2. Implement the `generate()` and `get_provider_name()` methods
3. Add it to `src/providers/__init__.py`
4. Update `run_analysis.py` to support the new provider

Example:
```python
from src.providers.base import LLMProvider, LLMResponse

class MyNewProvider(LLMProvider):
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        # Your implementation
        pass
    
    def get_provider_name(self) -> str:
        return "my_provider"
```

## Adding New Pipeline Strategies

1. Create a new pipeline class in `src/pipelines/` that inherits from `Pipeline`
2. Implement the `analyze()` method
3. Add it to `src/pipelines/__init__.py`
4. Update `run_analysis.py` to support the new pipeline

## Results

Results are saved in the `results/` directory with:
- Individual JSON files for each CV/pipeline/model combination
- `summary.json` with experiment overview
- `comparison.csv` for easy analysis in Excel/Pandas

## Configuration

Edit `config.yaml` to customize:
- Available LLM models
- Pipeline settings
- Temperature and token limits
- Results directory

