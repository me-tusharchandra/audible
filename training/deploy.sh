#!/usr/bin/env bash
#
# Deploy the audible env to Hugging Face Spaces.
#
# Run AFTER:
#   - `openenv validate audible_env` returns OK
#   - the env's README.md has the right metadata frontmatter
#   - `hf auth whoami` confirms you're logged in
#
# Pushes the env from the local audible_env/ directory to the user's HF account.

set -e

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
OPENENV="${REPO_ROOT}/.venv/bin/openenv"
HF="${REPO_ROOT}/.venv/bin/hf"

USERNAME=$("${HF}" auth whoami 2>/dev/null | awk -F'=' '/user=/{print $2}')
if [ -z "${USERNAME}" ]; then
    echo "ERROR: not logged in to Hugging Face. Run: hf auth login"
    exit 1
fi

REPO_ID="${1:-${USERNAME}/audible-env}"

echo "==[ deploying env to HF Space ${REPO_ID} ]======================="

echo "1. Validating env structure..."
"${OPENENV}" validate audible_env

echo
echo "2. Pushing to ${REPO_ID}..."
cd audible_env
"${OPENENV}" push --repo-id "${REPO_ID}"
cd ..

SPACE_URL="https://huggingface.co/spaces/${REPO_ID}"
DOCKER_REGISTRY=$(echo "${REPO_ID}" | sed 's:/:-:')
echo
echo "==[ deployed ]===================================================="
echo "Space:   ${SPACE_URL}"
echo "Docker:  registry.hf.space/${DOCKER_REGISTRY}:latest"
echo
echo "First build can take ~5 min on HF. Check status at:"
echo "  ${SPACE_URL}"
