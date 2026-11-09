# End-to-end reproduction of the final submission (Windows PowerShell).  ~6-7 h on one 12 GB GPU.
#   .\run_all.ps1 -DataRoot "C:\path\to\Project_2_Dataset"
# Every stage is resumable: trained models (DONE markers) and cached outputs are skipped on a re-run.
param([Parameter(Mandatory = $true)][string]$DataRoot)
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
function Step([string[]]$StepArgs) {
    Write-Host ">>> python run_pipeline.py $($StepArgs -join ' ')" -ForegroundColor Cyan
    & python -W ignore run_pipeline.py @StepArgs --data-root $DataRoot
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $($StepArgs -join ' ')" -ForegroundColor Red; exit 1 }
}

Step @("--stage", "prepare", "--config", "configs/final.json")
Step @("--stage", "cp_train", "--config", "configs/family_ft1.json", "--modality", "iv", "ex", "--folds", "0", "1", "2", "3")
Step @("--stage", "cp_train", "--config", "configs/family_ft6.json", "--modality", "iv", "--folds", "0", "1", "2", "3", "4")
Step @("--stage", "cp_infer", "--config", "configs/family_ft1.json", "--modality", "iv", "ex", "--folds", "0", "1", "2", "3")
Step @("--stage", "cp_infer", "--config", "configs/family_ft6.json", "--modality", "iv", "--folds", "0", "1", "2", "3", "4")
Step @("--stage", "cp_blend", "--config", "configs/final.json", "--modality", "iv", "--tags", "ft1", "ft6", "--out-tag", "ftmix6")
Step @("--stage", "submit", "--config", "configs/final.json", "--out", "submissions/submission.csv")
Write-Host "DONE: submissions\submission.csv" -ForegroundColor Green
