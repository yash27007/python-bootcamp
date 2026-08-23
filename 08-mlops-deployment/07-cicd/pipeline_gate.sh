#!/usr/bin/env bash
# pipeline_gate.sh -- a from-scratch, manual CI/CD "pipeline": three gated
# stages chained by shell exit codes. Each stage only runs if the previous
# one succeeded. This is the mechanism a real CI system (07-cicd's GitHub
# Actions YAML) automates and triggers on a Git event -- here it's run by
# hand to make the DAG-of-gated-stages idea concrete before the YAML.
#
# Usage:
#   pipeline_gate.sh <path-to-pipeline-dir>
#
# <path-to-pipeline-dir> must contain pipeline.py and test_pipeline.py
# (the same layout as 08-mlops-deployment/03-testing-ci).

set -uo pipefail

PYTEST="/home/yashwanth-aravind/ml-course/python-bootcamp/.venv/bin/pytest"
TARGET_DIR="${1:?usage: pipeline_gate.sh <path-to-pipeline-dir>}"

echo "=================================================================="
echo " STAGE 1/3: TEST  (gate: pytest exit code)"
echo "=================================================================="
"$PYTEST" "$TARGET_DIR/test_pipeline.py" -v
TEST_EXIT=$?

if [ "$TEST_EXIT" -ne 0 ]; then
    echo
    echo "=================================================================="
    echo " PIPELINE HALTED at STAGE 1 (tests failed, exit code $TEST_EXIT)"
    echo " -> build and deploy stages were never reached."
    echo "=================================================================="
    exit 1
fi

echo
echo "=================================================================="
echo " STAGE 2/3: BUILD  (gate: tests passed)"
echo "=================================================================="
echo "  packaging pipeline.py + trained model artifact ..."
echo "  build OK"

echo
echo "=================================================================="
echo " STAGE 3/3: DEPLOY GATE  (gate: build succeeded)"
echo "=================================================================="
echo "  all gates passed -> would deploy"

echo
echo "=================================================================="
echo " PIPELINE SUCCEEDED: test -> build -> would-deploy"
echo "=================================================================="
exit 0
