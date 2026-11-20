#!/usr/bin/env bash
set -u
cd ~/miragebench
LOG(){ echo "[$(date '+%F %H:%M:%S')] $*"; }

for fac in no_light_random no_bg_variation no_pose_scale_random low_texture_realism; do
  if [ -f "data/synth/$fac/generation_summary.json" ]; then
    LOG "synth $fac var"
  else
    LOG "synth $fac uretiliyor"
    blender --background --python tools/blender_launch.py -- --out "data/synth/$fac" --n 3000 --width 1280 --height 720 --factor "$fac" || LOG "UYARI $fac"
  fi
done

SYNTH=()
for d in data/synth/*/; do
  SYNTH+=("$(basename "$d")=$d")
done
LOG "synth: ${SYNTH[*]}"

for arm in configs/arms/*.yaml; do
  LOG "KOL: $arm"
  python -m miragebench.experiment.cli --base configs/base_study.yaml --arm "$arm" --real-root data/real --synth "${SYNTH[@]}" --results results/study || LOG "UYARI $arm durdu"
done
LOG "=== BITTI ==="
