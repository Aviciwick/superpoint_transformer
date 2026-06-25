#!/usr/bin/env bash
set -euo pipefail

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

download_to_shell() {
  local url="$1"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url"
  else
    wget -qO- "$url"
  fi
}

install_tinytex() {
  echo "Installing TinyTeX into $TINYTEX_DIR"
  download_to_shell "https://tinytex.yihui.org/install-unx.sh" | env TINYTEX_DIR="$TINYTEX_DIR" sh
}

main() {
  local bin_dir
  if ! bin_dir="$(find_tinytex_bin_dir)"; then
    install_tinytex
    bin_dir="$(find_tinytex_bin_dir)"
  fi

  export PATH="$bin_dir:$PATH"

  tlmgr option repository https://mirror.ctan.org/systems/texlive/tlnet >/dev/null
  tlmgr update --self || true
  tlmgr install \
    latexmk \
    ieeetran \
    algorithms \
    subfig \
    sttools \
    cite \
    hypcap \
    grfext

  echo "TinyTeX bin: $bin_dir"
  latexmk -v | sed -n '1p'
  pdflatex --version | sed -n '1p'
  bibtex --version | sed -n '1p'
}

main "$@"
