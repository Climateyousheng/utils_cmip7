#!/bin/bash
#
# Extract annual means from raw monthly files and create plots
#
# Usage: ./extract_raw.sh EXPERIMENT [OUTPUT_DIR]
#   EXPERIMENT  - Experiment name (e.g., xqhuj, xqhuk)
#   OUTPUT_DIR  - Output directory for plots (default: ./plots)
#
# Example: ./extract_raw.sh xqhuk ./my_plots
#

set -e  # Exit on error

# Parse arguments
EXPT="${1}"
OUTDIR="${2:-./plots}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if experiment name provided
if [ -z "$EXPT" ]; then
    echo "ERROR: Experiment name required"
    echo ""
    echo "Usage: $0 EXPERIMENT [OUTPUT_DIR]"
    echo ""
    echo "Examples:"
    echo "  $0 xqhuj"
    echo "  $0 xqhuk ./plots"
    echo ""
    exit 1
fi

echo "========================================================================"
echo "Extract Annual Means for ${EXPT}"
echo "========================================================================"
echo "Script directory: ${SCRIPT_DIR}"
echo "Output directory: ${OUTDIR}"
echo ""

# Check if we're on bp1 (has iris environment)
if [ -d "$HOME/iris" ]; then
    echo "Detected bp1 environment"
    echo "Activating iris environment..."
    source ~/iris/bin/activate
elif command -v conda &> /dev/null; then
    echo "Detected conda environment"
    # Activate conda environment if needed
    # conda activate your_env_name
else
    echo "Using system Python"
fi

echo ""
echo "Python version:"
python --version
echo ""

# Create output directory
mkdir -p "${OUTDIR}"

# Run extraction
echo "========================================================================"
echo "Running extraction..."
echo "========================================================================"
echo ""

python "${SCRIPT_DIR}/extract_raw.py" \
    "${EXPT}" \
    --outdir "${OUTDIR}" \
    --base-dir ~/dump2hold

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "SUCCESS!"
    echo "========================================================================"
    echo "Plots saved to: ${OUTDIR}/"
    echo ""
    echo "Generated files:"
    ls -lh "${OUTDIR}/${EXPT}"*.png 2>/dev/null || echo "  (No plots found - check for errors above)"
    echo "========================================================================"
else
    echo ""
    echo "========================================================================"
    echo "FAILED with exit code ${EXIT_CODE}"
    echo "========================================================================"
    exit $EXIT_CODE
fi
