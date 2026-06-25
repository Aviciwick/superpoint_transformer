#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PAPER_DIR="${PAPER_DIR:-$REPO_ROOT/期刊论文撰写/bsr_spt}"
MAIN_TEX="${1:-bsr_spt_tgrs.tex}"
BUILD_DIR="$PAPER_DIR/build"

if [[ "$(uname)" == "Darwin" ]]; then
  DEFAULT_TINYTEX_DIR="$HOME/Library/TinyTeX"
else
  DEFAULT_TINYTEX_DIR="$HOME/.TinyTeX"
fi

TINYTEX_DIR="${TINYTEX_DIR:-$DEFAULT_TINYTEX_DIR}"

find_tinytex_bin_dir() {
  local candidate
  shopt -s nullglob
  for candidate in "$TINYTEX_DIR"/bin/*; do
    if [[ -x "$candidate/latexmk" && -x "$candidate/pdflatex" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

main() {
  local bin_dir
  if ! bin_dir="$(find_tinytex_bin_dir)"; then
    "$SCRIPT_DIR/setup_latex_env.sh"
    bin_dir="$(find_tinytex_bin_dir)"
  fi

  export PATH="$bin_dir:$PATH"

  if [[ ! -f "$PAPER_DIR/$MAIN_TEX" ]]; then
    echo "LaTeX entry file not found: $PAPER_DIR/$MAIN_TEX" >&2
    exit 1
  fi

  mkdir -p "$BUILD_DIR"

  (
    cd "$PAPER_DIR"
    latexmk \
      -pdf \
      -interaction=nonstopmode \
      -file-line-error \
      -halt-on-error \
      -synctex=1 \
      -outdir=build \
      "$MAIN_TEX"
  )

  echo "PDF generated at: $BUILD_DIR/${MAIN_TEX%.tex}.pdf"
}

main "$@"
