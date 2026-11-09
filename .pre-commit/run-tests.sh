#!/usr/bin/env bash
set -e

echo "==> Running pytest suite..."

if command -v uv >/dev/null 2>&1; then
    uv run pytest
else
    pytest
fi

