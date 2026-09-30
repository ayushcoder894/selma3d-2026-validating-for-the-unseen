#!/usr/bin/env bash
# Package weights into submission/resources, build the Grand Challenge container, test it on the
# volumes in submission/test/input/images/<interface>/ (if any) and save it as a .tar.gz.
#
#   bash scripts/build_container.sh v15    v15a.pt v15b.pt                              # Task 2 entry
#   bash scripts/build_container.sh routed v15a.pt v15b.pt -- B128_all1.pt B128_all2.pt  # Task 1 entry
#
# Checkpoints may be training checkpoints (runs/*/final_ema.pt) or the released fp16 weights.
set -eu
cd "$(dirname "$0")/.."
NAME=${1:?name}; shift
iso=() cont=() side=iso
for c in "$@"; do
  [ "$c" = "--" ] && { side=cont; continue; }
  [ -f "$c" ] || { echo "missing checkpoint $c"; exit 1; }
  if [ $side = iso ]; then iso+=("$c"); else cont+=("$c"); fi
done
[ ${#iso[@]} -gt 0 ] || { echo "need at least one 96^3 checkpoint"; exit 1; }
RES=submission/resources

echo "== package isolated group (${iso[*]})"
python selma/package.py --resources "$RES" --clean --checkpoint "${iso[@]}" --threshold 0.45 \
    --minimum-component-size 4 --sparse-threshold 0.3 --sparse-minimum-component-size 8
rm -f "$RES"/cont/*.pt
if [ ${#cont[@]} -gt 0 ]; then
  echo "== package contiguous experts (${cont[*]})"
  mkdir -p "$RES/cont"
  python selma/package.py --resources "$RES/cont" --clean --checkpoint "${cont[@]}"
  cp submission/route.json "$RES/cont/route.json"
else
  rm -f "$RES/cont/route.json"          # no experts: every volume takes the v15 path
fi

echo "== docker build"
docker build --platform linux/amd64 -t "selma3d-$NAME" submission

if ls submission/test/input/images/*/*.mha > /dev/null 2>&1; then
  echo "== container test"
  rm -rf submission/test/output && mkdir -p submission/test/output
  # (Git Bash on Windows: prefix with MSYS_NO_PATHCONV=1 so the volume paths are not rewritten)
  docker run --rm --gpus all --memory 16g --network none \
      -v "$PWD/submission/test/input:/input:ro" -v "$PWD/submission/test/output:/output" "selma3d-$NAME"
  ls -la submission/test/output/images/*/
fi

echo "== save"
docker save "selma3d-$NAME" | gzip -c > "selma3d-$NAME.tar.gz"
gzip -t "selma3d-$NAME.tar.gz" && ls -la "selma3d-$NAME.tar.gz"
