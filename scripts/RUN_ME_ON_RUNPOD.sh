#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
bash ./runpod_train_persona_lora.sh
