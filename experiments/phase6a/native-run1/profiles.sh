#!/usr/bin/env bash
# Phase 6A profiles: one configuration per process (moonlight_profile.py), cold and warm, alternating backends, then
# repeats of the two key configurations for run-to-run noise. Nothing else may run on the machine meanwhile.
# Run from anywhere: bash experiments/phase6a/native-run1/profiles.sh (existing outputs are skipped).
set -u
cd "$(dirname "$0")/../../.." || exit 1   # the repository root
RUN=${RUN:-experiments/phase6a/native-run1}
CFG=configs/phase6a-native.yaml
LOGDIR=${LOGDIR:-${TMPDIR:-/tmp}/phase6a-profile-logs}
PY=${PY:-python}   # an interpreter with uv installed
mkdir -p "$LOGDIR"
export PYTHONHASHSEED=1 PYTHONIOENCODING=utf-8
# name:mode:trace:tag
SPECS=(
  "python-stream:cold:trace:"
  "native-stream:cold:trace:"
  "python-hotness-80:cold::"
  "native-host-12g:cold:trace:"
  "python-stream:warm::"
  "native-stream:warm::"
  "python-hotness-80:warm::"
  "native-host-4g:warm::"
  "native-host-8g:warm::"
  "native-host-12g:warm:trace:"
  "native-host-12g-prefetch:warm::"
  "native-host-12g-freeze:warm::"
  "native-host-4g:cold::"
  "native-host-8g:cold::"
  "native-host-12g-freeze:cold::"
  "native-host-12g-prefetch:cold::"
  "python-stream:warm::-repeat"
  "native-host-12g:warm::-repeat"
)
for spec in "${SPECS[@]}"; do
  IFS=: read -r name mode trace tag <<< "$spec"
  flags=()
  [ "$mode" = "warm" ] && flags+=(--warm)
  [ "$trace" = "trace" ] && flags+=(--trace)
  out="$RUN/profile-$name-$mode$tag.json"
  if [ -f "$out" ]; then echo "skip $out"; continue; fi
  started=$(date +%s)
  "$PY" -m uv run python benchmarks/moonlight_profile.py --run "$RUN" --config "$CFG" --configuration "$name" \
      --prompts 7 0 3 5 "${flags[@]}" --output "$out" > "$LOGDIR/$name-$mode$tag.log" 2>&1
  code=$?
  echo "$(date +%H:%M:%S) $name $mode$tag exit $code after $(( $(date +%s) - started ))s"
  sleep 5
done
echo "profiles done"
