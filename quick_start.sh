#!/bin/bash

echo "=========================================="
echo "Probabilistic FedFeatGen Quick Start"
echo "=========================================="
echo ""

# Check if requirements are installed
echo "Checking dependencies..."
python3 -c "import torch; import transformers; import sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install -r requirements_federated.txt
fi

# Run tests
echo ""
echo "Running tests..."
python3 test_setup.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Tests failed! Please check the errors above."
    exit 1
fi

echo ""
echo "✅ All tests passed!"
echo ""
echo "=========================================="
echo "Choose an option:"
echo "=========================================="
echo "1) Train Probabilistic FIN (YOUR METHOD)"
echo "2) Train Original FIN (BASELINE)"
echo "3) Run Full Comparison (4 experiments)"
echo "4) Exit"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "Training Probabilistic FIN with UQ-weighted aggregation..."
        python3 train_federated.py \
            --use_probabilistic \
            --use_uq_weighting \
            --num_rounds 30 \
            --local_epochs 3 \
            --batch_size 32 \
            --lr 1e-4 \
            --uq_alpha 0.5 \
            --eval_every 5 \
            --save_dir checkpoints_prob
        ;;
    2)
        echo ""
        echo "Training Original FIN (Baseline)..."
        python3 train_federated.py \
            --num_rounds 30 \
            --local_epochs 3 \
            --batch_size 32 \
            --lr 1e-4 \
            --eval_every 5 \
            --save_dir checkpoints_det
        ;;
    3)
        echo ""
        echo "Running all comparison experiments..."
        echo "This will take a while (4 full training runs)..."
        python3 run_comparison.py
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice!"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo ""
echo "Check results in the checkpoint directories"
echo "- Probabilistic: checkpoints_prob/results.json"
echo "- Deterministic: checkpoints_det/results.json"
echo ""
