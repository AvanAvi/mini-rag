# Sets project-local caches for the current PowerShell session only.
# Usage (from the project root):  . .\scripts\env.ps1
# Writes nothing to the registry, PATH, or $PROFILE.

$s_project_root = Split-Path -Parent $PSScriptRoot
$s_fastembed_cache = Join-Path $s_project_root '.cache\fastembed'
$s_hf_home = Join-Path $s_project_root '.cache\huggingface'

New-Item -ItemType Directory -Force -Path $s_fastembed_cache | Out-Null
$env:FASTEMBED_CACHE_PATH = $s_fastembed_cache

# huggingface_hub (used by fastembed) otherwise writes to ~\.cache\huggingface
New-Item -ItemType Directory -Force -Path $s_hf_home | Out-Null
$env:HF_HOME = $s_hf_home

Write-Host "FASTEMBED_CACHE_PATH = $s_fastembed_cache"
Write-Host "HF_HOME = $s_hf_home"
