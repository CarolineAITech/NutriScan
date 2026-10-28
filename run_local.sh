#!/usr/bin/env bash
# Launches JupyterLab from the project folder using the local environment.
set -euo pipefail
cd "$(dirname "$0")"
exec "${HOME}/nutriscan-env/bin/jupyter" lab NutriScan_Milestone1.ipynb
