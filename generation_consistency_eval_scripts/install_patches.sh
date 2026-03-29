#!/bin/bash
# install_patches.sh
#
# Copies WR-Arena's modified WorldScore files into the WorldScore submodule.
# Run once after: git submodule update --init thirdparty/WorldScore
#                 pip install -e thirdparty/WorldScore

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS_ROOT="${PROJECT_ROOT}/thirdparty/WorldScore"

if [ ! -d "${WS_ROOT}" ]; then
    echo "Error: thirdparty/WorldScore not found."
    echo "Run: git submodule update --init thirdparty/WorldScore"
    exit 1
fi

PATCHES_DIR="${SCRIPT_DIR}/worldscore_patches"

echo "Installing WR-Arena patches into ${WS_ROOT} ..."

# Modified evaluator (per-round evaluation without VFIMamba)
cp "${PATCHES_DIR}/evaluator_per_round_arif.py" \
   "${WS_ROOT}/worldscore/benchmark/helpers/evaluator_per_round_arif.py"
echo "  copied evaluator_per_round_arif.py"

echo "Done. Patches installed successfully."
echo ""
echo "Next steps:"
echo "  1. Follow WorldScore's setup for DROID-SLAM, GroundingDINO, and SAM2"
echo "     if you want camera_control, 3d_consistency, and object_control aspects."
echo "  2. Generate videos:  bash generation_consistency_eval_scripts/pan.sh"
echo "  3. Prepare dirs:     python generation_consistency_eval_scripts/prepare_worldscore_dirs.py ..."
echo "  4. Evaluate:         python generation_consistency_eval_scripts/run_evaluate_multiround.py ..."
