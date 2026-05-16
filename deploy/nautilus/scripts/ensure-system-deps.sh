#!/usr/bin/env bash
# Idempotent apt packages for stock pytorch/pytorch pods (git, ssh-keygen not in the image).
# Runs at pod start; safe to call from every Deployment/Pod entrypoint.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

need_apt=0
for cmd in git; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    need_apt=1
    break
  fi
done

if [[ "${need_apt}" -eq 0 ]]; then
  exit 0
fi

echo "Installing system packages (git, openssh-client) …" >&2
apt-get update -qq
apt-get install -y -qq --no-install-recommends git ca-certificates openssh-client
rm -rf /var/lib/apt/lists/*
