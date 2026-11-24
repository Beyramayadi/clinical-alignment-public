# Clinical Alignment Workspace

This workspace contains scripts for generating and enriching synthetic menstrual-health data, plus a utility API connectivity check.

## Project Files

- [script.py](script.py): Generates synthetic menstrual-health dataset JSON.
- [enrich_first_sample_note.py](enrich_first_sample_note.py): Calls Azure OpenAI to generate a note for the first sample.
- [api_call.py](api_call.py): Quick check that your Azure OpenAI endpoint/key are working and lists available models.
- [menstrual_health_dataset.json](menstrual_health_dataset.json): Main generated dataset.
- [menstrual_health_dataset_first_note_enriched.json](menstrual_health_dataset_first_note_enriched.json): Output after note enrichment.
- [data.ipynb](data.ipynb): Notebook for exploration.

## Reproducible Environment (New Machine)

Use the bootstrap script to recreate the same project environment, including notebook dependencies and Jupyter kernel registration.

### 1) Run bootstrap once

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_env.ps1
```

This will:

- Create `.venv` with Python 3.10
- Install pinned dependencies from `requirements.txt`
- Install spaCy models used in notebooks (`en_core_web_sm` and `en_core_sci_sm`)
- Register a Jupyter kernel named `Python (clinical-alignment)`

### 2) Activate environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3) In notebooks

Select kernel: `Python (clinical-alignment)`

## Requirements

- Python 3.10+
- Local virtual environment at `.venv`

## Setup

### 1) Activate virtual environment (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2) Configure environment variables

Use local file [.env](.env) for secret storage (already ignored by [.gitignore](.gitignore)).

Expected format:

```env
AZURE_OPENAI_API_KEY=your_real_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/

# Optional aliases
API_KEY=your_real_key
API_ENDPOINT=https://your-resource-name.openai.azure.com/
```

You can also set variables directly in PowerShell for the current shell session:

```powershell
$env:AZURE_OPENAI_API_KEY="your_real_key"
$env:AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/"
```

## Usage

### Generate dataset

```powershell
python .\script.py
```

### Test API connectivity and list models

```powershell
python .\api_call.py
```

### Enrich first sample with generated note

```powershell
python .\enrich_first_sample_note.py
```

## Security Notes

- Never commit real keys or endpoints in source files.
- Keep real secrets only in [.env](.env) or shell environment variables.
- Rotate any key that was previously hardcoded in code history.
