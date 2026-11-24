param(
    [string]$PythonVersion = "3.10",
    [string]$VenvDir = ".venv",
    [string]$KernelName = "clinical-alignment",
    [string]$KernelDisplayName = "Python (clinical-alignment)"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "[1/6] Creating virtual environment at $VenvDir"
if (-not (Test-Path $VenvDir)) {
    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $PyLauncher) {
        py -$PythonVersion -m venv $VenvDir
    } else {
        $PythonCmd = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $PythonCmd) {
            throw "Neither 'py' nor 'python' command is available. Install Python first."
        }
        python -m venv $VenvDir
    }
} else {
    Write-Host "Virtual environment already exists, reusing it."
}

$VenvPython = Join-Path $RepoRoot "$VenvDir\bin\python"
if (-not (Test-Path $VenvPython)) {
    throw "Could not find venv Python at $VenvPython"
}

Write-Host "[2/6] Upgrading pip tooling"
& $VenvPython -m pip install --upgrade pip setuptools wheel

Write-Host "[3/6] Installing pinned project dependencies"
& $VenvPython -m pip install -r requirements.txt

Write-Host "[4/6] Installing spaCy models for notebooks"
& $VenvPython -m spacy download en_core_web_sm

# SciSpaCy model used by readability_metrics.ipynb
& $VenvPython -m pip install "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"

Write-Host "[5/6] Registering Jupyter kernel"
& $VenvPython -m ipykernel install --user --name $KernelName --display-name $KernelDisplayName

Write-Host "[6/6] Done"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1) Activate env: .\\$VenvDir\\Scripts\\Activate.ps1"
Write-Host "  2) Open notebook and select kernel: $KernelDisplayName"
