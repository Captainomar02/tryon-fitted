#!/usr/bin/env bash
set -euo pipefail

# This file is run on instance start. Output in /var/log/onstart.log
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/scripts/vast/onstart.sh"
