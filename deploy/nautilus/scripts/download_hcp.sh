#!/usr/bin/env bash
# Download HCP 1200 minimally preprocessed T1w volumes from s3://hcp-openaccess.
# Requires AWS CLI + ConnectomeDB HCP credentials (~/.aws on PVC; already set up on NRP).
#
# Example:
# bash /files/repo/uncertainty-quantification/deploy/nautilus/scripts/download_hcp.sh
#
# Parallel (subjects in flight; tune down if S3 throttles):
# PARALLEL_JOBS=8 bash /files/repo/uncertainty-quantification/deploy/nautilus/scripts/download_hcp.sh
#
# Smoke test (10 subjects, no S3 listing — uses bundled IDs in repo):
# SUBJECT_LIST_FILE=/files/repo/uncertainty-quantification/deploy/nautilus/scripts/hcp_subjects_test10.txt PARALLEL_JOBS=4 bash deploy/nautilus/scripts/download_hcp.sh
#
# Custom subset (one subject id per line):
# SUBJECT_LIST_FILE=/files/repo/uncertainty-quantification/datasets/hcp/my_subjects.txt bash deploy/nautilus/scripts/download_hcp.sh
#
# Re-list subjects from S3 (default: reuse datasets/hcp/.subjects.txt if present):
# REFRESH_SUBJECT_LIST=1 bash deploy/nautilus/scripts/download_hcp.sh
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
BUCKET="s3://hcp-openaccess/HCP_1200"
OUTDIR="${HCP_OUTDIR:-/files/repo/uncertainty-quantification/datasets/hcp}"
SUBJECT_LIST="${OUTDIR}/.subjects.txt"
LOG="${OUTDIR}/download.log"
LOG_LOCK="${OUTDIR}/.download.log.lock"
FAIL_DIR="${OUTDIR}/.failures"
REFRESH_SUBJECT_LIST="${REFRESH_SUBJECT_LIST:-0}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"

T1_KEY="T1w/T1w_acpc_dc_restore_brain.nii.gz"
SEG_KEY="T1w/aparc+aseg.nii.gz"
MASK_KEY="T1w/brainmask_fs.nii.gz"

# Read 6-digit subject IDs (skip # comments, blank lines, CRLF).
# Use grep for {6} — POSIX awk (mawk in pytorch image) does not support {n} intervals.
filter_subject_ids() {
  sed 's/\r$//' |
    grep -vE '^[[:space:]]*#|^[[:space:]]*$' |
    sed 's/^[[:space:]]*//;s/[[:space:]]*$//' |
    grep -E '^[0-9]{6}$'
}

normalize_subject_list() {
  filter_subject_ids <"$1" >"$2"
}

log() {
  {
    flock -x 200
    echo "$*" | tee -a "$LOG"
  } 200>"$LOG_LOCK"
}

mark_failure() {
  local marker="$1"
  mkdir -p "$FAIL_DIR"
  touch "${FAIL_DIR}/${marker}"
}

pull_object() {
  local subj="$1"
  local rel_key="$2"
  local dest="${OUTDIR}/${subj}/${rel_key}"
  local src="${BUCKET}/${subj}/${rel_key}"

  if [[ -s "$dest" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  if aws s3 cp "$src" "$dest" --region "$AWS_REGION" --only-show-errors; then
    return 0
  fi
  log "WARN: failed ${src} -> ${dest}"
  mark_failure "${subj}_$(basename "$dest")"
  return 1
}

download_subject() {
  local subj="$1"

  if [[ ! "$subj" =~ ^[0-9]{6}$ ]]; then
    log "WARN: skipping invalid subject id: ${subj}"
    return 0
  fi

  log "Downloading ${subj}"

  pull_object "$subj" "$T1_KEY" || true
  pull_object "$subj" "$SEG_KEY" || true
  pull_object "$subj" "$MASK_KEY" || true
}

export AWS_REGION BUCKET OUTDIR T1_KEY SEG_KEY MASK_KEY LOG LOG_LOCK FAIL_DIR
export -f log mark_failure pull_object download_subject

mkdir -p "$OUTDIR"
rm -rf "$FAIL_DIR"
mkdir -p "$FAIL_DIR"
: >"$LOG"

command -v aws >/dev/null 2>&1 || {
  echo "ERROR: aws CLI not found. Install awscli in the pod." >&2
  exit 1
}

log "Checking AWS credentials (region=${AWS_REGION})..."
aws sts get-caller-identity --region "$AWS_REGION" >>"$LOG" 2>&1

ACTIVE_LIST="${OUTDIR}/.subjects.active.txt"

if [[ -n "${SUBJECT_LIST_FILE:-}" ]]; then
  if [[ ! -f "$SUBJECT_LIST_FILE" ]]; then
    echo "ERROR: SUBJECT_LIST_FILE not found: ${SUBJECT_LIST_FILE}" >&2
    exit 1
  fi
  normalize_subject_list "$SUBJECT_LIST_FILE" "$ACTIVE_LIST"
  SUBJECT_LIST="$ACTIVE_LIST"
  log "Using subject list: ${SUBJECT_LIST_FILE} -> ${SUBJECT_LIST}"
elif [[ "$REFRESH_SUBJECT_LIST" == "1" || ! -s "$SUBJECT_LIST" ]]; then
  log "Listing subjects under ${BUCKET}/ ..."
  aws s3 ls "${BUCKET}/" --region "$AWS_REGION" |
    awk '/PRE/ { gsub(/\//, "", $NF); print $NF }' |
    filter_subject_ids |
    sort -u >"$SUBJECT_LIST"
  log "Wrote ${SUBJECT_LIST}"
  cp "$SUBJECT_LIST" "$ACTIVE_LIST"
  SUBJECT_LIST="$ACTIVE_LIST"
else
  log "Reusing cached subject list: ${SUBJECT_LIST} (set REFRESH_SUBJECT_LIST=1 to rebuild)"
  cp "$SUBJECT_LIST" "$ACTIVE_LIST"
  SUBJECT_LIST="$ACTIVE_LIST"
fi

N_SUBJ="$(wc -l <"$SUBJECT_LIST" | tr -d ' ')"
if [[ "$N_SUBJ" -lt 1 ]]; then
  echo "ERROR: no 6-digit subject IDs in ${SUBJECT_LIST}" >&2
  if [[ -n "${SUBJECT_LIST_FILE:-}" && -f "${SUBJECT_LIST_FILE}" ]]; then
    echo "  Source file (${SUBJECT_LIST_FILE}) has $(wc -l <"${SUBJECT_LIST_FILE}" | tr -d ' ') line(s). First lines:" >&2
    head -5 "${SUBJECT_LIST_FILE}" | sed 's/^/    /' >&2
    echo "  IDs look valid but were not parsed — update download_hcp.sh (filter_subject_ids fix)." >&2
  fi
  exit 1
fi

log "Sample subjects: $(head -3 "$SUBJECT_LIST" | tr '\n' ' ')..."
log "Total subjects: ${N_SUBJ}"
log "Output: ${OUTDIR}"
log "Parallel jobs: ${PARALLEL_JOBS} (set PARALLEL_JOBS=8 or 16 to go faster)"

if [[ "$PARALLEL_JOBS" -le 1 ]]; then
  while IFS= read -r subj || [[ -n "$subj" ]]; do
    [[ -z "$subj" ]] && continue
    download_subject "$subj"
  done <"$SUBJECT_LIST"
else
  xargs -P "$PARALLEL_JOBS" -I{} bash -c 'download_subject "$@"' _ {} <"$SUBJECT_LIST"
fi

FAILURES="$(find "$FAIL_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')"
if [[ "$FAILURES" -gt 0 ]]; then
  log "Done with ${FAILURES} failed download(s). See ${LOG}"
  exit 1
fi

log "Done. All files present under ${OUTDIR}"
