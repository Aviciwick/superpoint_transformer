#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PAPER_DIR="${PAPER_DIR:-$REPO_ROOT/期刊论文撰写/bsr_spt}"
BUILD_DIR="$PAPER_DIR/build"

rm -rf "$BUILD_DIR"
echo "Removed $BUILD_DIR"
