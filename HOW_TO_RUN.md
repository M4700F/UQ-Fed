# Complete Guide: How to Run Probabilistic FedFeatGen

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Verify Setup](#verify-setup)
4. [Running Experiments](#running-experiments)
5. [Understanding the Output](#understanding-the-output)
6. [Analyzing Results](#analyzing-results)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)
9. [FAQ](#faq)

---

## 🔧 Prerequisites

### System Requirements

- **OS**: Linux, macOS, or Windows (with WSL)
- **Python**: 3.8 or higher
- **GPU**: CUDA-capable GPU recommended (CPU works but slower)
- **RAM**: Minimum 16GB (32GB recommended)
- **Disk Space**: ~10GB for models and data

### Data Requirements

Your data should be organized as:

```
pcfi/
├── Fed_Data/
│   ├── images/
│   │   ├── chexpert/
│   │   └── nih/
│   ├── reports/
│   │   └── nih/
│   └── splits/
│       ├── client_0.csv
│       ├── client_1.csv
│       └── ... (client_9.csv)
└── PadChest_Data/
    ├── images/
    └── padchest_test.csv
```

**✅ You already have this setup!**

---

## 📦 Installation

### Step 1: Check Python Version

```bash
python3 --version
```

Should output: `Python 3.8.x` or higher

### Step 2: Install Dependencies

```bash
cd /Users/nafisfuadshahid/Documents/Paper\ Implementation/Confidence-Based\ Feature\ Imputation/pcfi

pip install -r requirements_federated.txt
```

**What this installs:**
- `torch` - PyTorch for deep learning
- `torchvision` - Image models (ResNet-50)
- `transformers` - BERT text encoder
- `scikit-learn` - Evaluation metrics
- `pandas`, `numpy` - Data processing
- `Pillow` - Image loading
- `tqdm` - Progress bars
- `matplotlib` - Plotting

### Step 3: Download Pre-trained Models (Automatic)

The first time you run, it will automatically download:
- **ResNet-50** (~100MB)
- **BERT-base** (~440MB)

This happens on first run, so be patient!

---

## ✅ Verify Setup

### Run Test Script

```bash
python3 test_setup.py
```

**Expected Output:**
```
============================================================
TESTING PROBABILISTIC FEDFEATGEN IMPLEMENTATION
============================================================

Testing imports...
✓ probabilistic_models imported successfully
✓ multimodal_model imported successfully
✓ federated_learning imported successfully
✓ data_loader_federated imported successfully

Testing Probabilistic FIN...
✓ Probabilistic FIN working correctly
  Alpha range: [0.266, 2.029]
  Beta range: [0.258, 2.592]
  Loss: 0.1528
  Mean uncertainty: 0.091368

Testing Multimodal Model...
✓ Multimodal Model working correctly

Testing Federated Aggregation...
✓ Federated Aggregation working correctly

============================================================
✓ ALL TESTS PASSED!
============================================================
```

**If tests fail**, see [Troubleshooting](#troubleshooting) section.

---

## 🚀 Running Experiments

### Method 1: Interactive Quick Start (Easiest!)

```bash
./quick_start.sh
```

This gives you a menu:
```
==========================================
Choose an option:
==========================================
1) Train Probabilistic FIN (YOUR METHOD)
2) Train Original FIN (BASELINE)
3) Run Full Comparison (4 experiments)
4) Exit
```

**Recommended:** Start with option **1** for a single experiment.

### Method 2: Direct Commands

#### A. Train YOUR METHOD (Probabilistic FIN + UQ)

```bash
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
```

**What these parameters mean:**
- `--use_probabilistic`: Use Beta-NLL loss (your method)
- `--use_uq_weighting`: Weight clients by uncertainty
- `--num_rounds 30`: 30 federated communication rounds
- `--local_epochs 3`: 3 epochs of local training per round
- `--batch_size 32`: 32 samples per batch
- `--lr 1e-4`: Learning rate
- `--uq_alpha 0.5`: Balance between data size and uncertainty (0.5 = equal)
- `--eval_every 5`: Evaluate on test set every 5 rounds
- `--save_dir`: Where to save checkpoints

#### B. Train BASELINE (Original FIN)

```bash
python3 train_federated.py \
    --num_rounds 30 \
    --local_epochs 3 \
    --batch_size 32 \
    --lr 1e-4 \
    --eval_every 5 \
    --save_dir checkpoints_det
```

**Note:** No `--use_probabilistic` or `--use_uq_weighting` flags = baseline method

#### C. Quick Test Run (5 rounds, 2 minutes)

```bash
python3 train_federated.py \
    --use_probabilistic \
    --use_uq_weighting \
    --num_rounds 5 \
    --local_epochs 1 \
    --batch_size 32 \
    --eval_every 1 \
    --save_dir checkpoints_test
```

**Use this to verify everything works before running full 30 rounds!**

### Method 3: Full Comparison (All 4 Experiments)

```bash
python3 run_comparison.py
```

**This runs:**
1. Probabilistic FIN + UQ-weighted FedAvg (YOUR METHOD)
2. Deterministic FIN + Standard FedAvg (BASELINE)
3. Probabilistic FIN + Standard FedAvg (Ablation 1)
4. Deterministic FIN + UQ-weighted FedAvg (Ablation 2)

**⚠️ Warning:** This takes 4× longer (runs 4 full experiments sequentially)!

**Estimated Time:**
- On GPU: ~2-4 hours per experiment = **8-16 hours total**
- On CPU: ~8-12 hours per experiment = **32-48 hours total**

---

## 📊 Understanding the Output

### During Training

You'll see output like this:

```
============================================================
Round 1/30
============================================================

Client 0 (Multimodal):
  Uncertainty: 0.045234, Data size: 1152

Client 1 (Multimodal):
  Uncertainty: 0.038912, Data size: 1163

...

Client 5 (Unimodal):
  Uncertainty: 0.067543, Data size: 2748

...

--- Aggregating models ---

Client weights (data + UQ):
  Client 0: data_wt=0.115, conf=0.105, final_wt=0.110, unc=0.0452
  Client 1: data_wt=0.116, conf=0.107, final_wt=0.112, unc=0.0389
  Client 2: data_wt=0.117, conf=0.104, final_wt=0.111, unc=0.0523
  ...

--- Evaluating on test set ---
Evaluating: 100%|████████████████| 50/50 [00:45<00:00]
Test AUC: 0.8734
Test F1: 0.7612

Saved checkpoint to checkpoints_prob/checkpoint_round_5.pth
```

### Key Metrics to Watch

1. **Uncertainty**: Lower = more confident
   - Multimodal clients: Usually 0.02-0.05 (have real data)
   - Unimodal clients: Usually 0.05-0.10 (using imputed features)

2. **Client weights**: Higher weight = more influence on global model
   - Good: Multimodal clients have higher weights
   - Your UQ weighting ensures this!

3. **Test AUC**: Main metric (higher is better)
   - Goal: > 0.86 (baseline)
   - Your method: Should reach 0.87-0.89

4. **Test F1**: Secondary metric (higher is better)
   - Goal: > 0.75 (baseline)
   - Your method: Should reach 0.76-0.78

### Output Files

After training, you'll have:

```
checkpoints_prob/
├── checkpoint_round_5.pth    # Model at round 5
├── checkpoint_round_10.pth   # Model at round 10
├── checkpoint_round_15.pth   # Model at round 15
├── checkpoint_round_20.pth   # Model at round 20
├── checkpoint_round_25.pth   # Model at round 25
├── checkpoint_round_30.pth   # Model at round 30
└── results.json              # All evaluation results
```

**results.json** contains:
```json
[
  {
    "round": 5,
    "auc": 0.8534,
    "f1": 0.7412
  },
  {
    "round": 10,
    "auc": 0.8712,
    "f1": 0.7589
  },
  ...
]
```

---

## 📈 Analyzing Results

### Step 1: Visualize Results

```bash
python3 visualize_results.py
```

**Output:**
```
Searching for experiment results...
✓ Found results in checkpoints_prob
✓ Found results in checkpoints_det

================================================================================
                         RESULTS SUMMARY
================================================================================
Method                                   Final AUC    Best AUC     Final F1     Best F1
--------------------------------------------------------------------------------
Prob                                     0.8789       0.8812       0.7645       0.7698
Det                                      0.8623       0.8654       0.7512       0.7556
================================================================================

================================================================================
                    IMPROVEMENT ANALYSIS
================================================================================

Probabilistic FIN vs Deterministic FIN:
  Final AUC improvement: +1.93%
  Best AUC improvement:  +1.83%
  Final F1 improvement:  +1.77%
  Best F1 improvement:   +1.88%
================================================================================

✅ Visualization complete!
```

**Generated Files:**
- `plot_prob.png` - Your method's learning curves
- `plot_det.png` - Baseline learning curves
- `comparison_all.png` - Side-by-side comparison

### Step 2: Check Individual Results

**View JSON directly:**
```bash
cat checkpoints_prob/results.json | python3 -m json.tool
```

**Or use Python:**
```python
import json

with open('checkpoints_prob/results.json', 'r') as f:
    results = json.load(f)

# Get best AUC
best = max(results, key=lambda x: x['auc'])
print(f"Best AUC: {best['auc']:.4f} at round {best['round']}")
```

### Step 3: Load Trained Model

```python
import torch
from multimodal_model import get_model
from probabilistic_models import ProbabilisticFIN

# Load checkpoint
checkpoint = torch.load('checkpoints_prob/checkpoint_round_30.pth')

# Create models
model = get_model(use_uncertainty=True)
fin_i2t = ProbabilisticFIN()

# Load weights
model.load_state_dict(checkpoint['model_state'])
fin_i2t.load_state_dict(checkpoint['fin_i2t_state'])

# Use for inference
model.eval()
fin_i2t.eval()
```

---

## 🔧 Troubleshooting

### Problem 1: Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
```bash
pip install -r requirements_federated.txt
```

### Problem 2: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution 1: Reduce batch size**
```bash
python3 train_federated.py --batch_size 16  # or even 8
```

**Solution 2: Use CPU**
```bash
# The code automatically uses CPU if CUDA is unavailable
# Just make sure no GPU is being used by other processes
```

**Solution 3: Reduce model size (advanced)**
Edit `multimodal_model.py`:
- Use `bert-base-uncased` → `distilbert-base-uncased`
- Smaller dimension: `feature_dim=128` instead of `256`

### Problem 3: Variance Explosion (NaN or inf)

**Error:**
```
RuntimeError: loss is nan
```

**Solution 1: Reduce learning rate**
```bash
python3 train_federated.py --lr 5e-5  # instead of 1e-4
```

**Solution 2: Increase variance penalty**

Edit `probabilistic_models.py`, line 36:
```python
self.beta_nll_loss = BetaNLLLoss(variance_penalty=0.05)  # was 0.01
```

**Solution 3: Tighter parameter bounds**

Edit `probabilistic_models.py`, line 28-31:
```python
BetaNLLLoss(
    min_alpha=0.5,    # was 0.1
    max_alpha=50.0,   # was 100.0
    min_beta=0.5,     # was 0.1
    max_beta=50.0     # was 100.0
)
```

### Problem 4: Slow Training

**Issue:** Training is taking too long

**Solution 1: Use GPU**
```bash
# Check if CUDA is available
python3 -c "import torch; print(torch.cuda.is_available())"
```

**Solution 2: Reduce dataset size (for testing)**

Edit `data_loader_federated.py`, add after line 85:
```python
# Limit dataset size for testing
if len(self.df) > 500:
    self.df = self.df.sample(500)
```

**Solution 3: Reduce number of rounds**
```bash
python3 train_federated.py --num_rounds 10  # instead of 30
```

### Problem 5: Data Not Found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'Fed_Data/splits/client_0.csv'
```

**Solution:** Ensure you're running from the correct directory
```bash
cd /Users/nafisfuadshahid/Documents/Paper\ Implementation/Confidence-Based\ Feature\ Imputation/pcfi

# Verify data exists
ls Fed_Data/splits/
ls PadChest_Data/
```

### Problem 6: BERT Download Fails

**Error:**
```
OSError: Can't load tokenizer for 'bert-base-uncased'
```

**Solution:** Manually download BERT
```python
from transformers import BertModel, BertTokenizer

# This will download to cache
BertModel.from_pretrained('bert-base-uncased')
BertTokenizer.from_pretrained('bert-base-uncased')
```

### Problem 7: Test Script Fails

**Error in test_setup.py**

**Solution:** Check specific error message
```bash
python3 test_setup.py 2>&1 | grep "Error"
```

Then refer to specific problem above based on error type.

---

## 🎓 Advanced Usage

### Custom Hyperparameters

#### Experiment with Different UQ Alpha

```bash
# More weight to uncertainty
python3 train_federated.py --use_probabilistic --use_uq_weighting --uq_alpha 0.8

# More weight to data size
python3 train_federated.py --use_probabilistic --use_uq_weighting --uq_alpha 0.2

# Only uncertainty (ignore data size)
python3 train_federated.py --use_probabilistic --use_uq_weighting --uq_alpha 1.0
```

#### Different Learning Rates

```bash
# Lower learning rate (more stable)
python3 train_federated.py --lr 5e-5

# Higher learning rate (faster, less stable)
python3 train_federated.py --lr 2e-4
```

#### More/Fewer Local Epochs

```bash
# More local training (better local models, more communication cost)
python3 train_federated.py --local_epochs 5

# Less local training (faster, may need more rounds)
python3 train_federated.py --local_epochs 1
```

### Ablation Studies

#### 1. Effect of Beta-NLL vs MSE

```bash
# With Beta-NLL (your method)
python3 train_federated.py --use_probabilistic --save_dir abl_betanll

# Without Beta-NLL (MSE)
python3 train_federated.py --save_dir abl_mse
```

#### 2. Effect of UQ-Weighted Aggregation

```bash
# With UQ weighting (your method)
python3 train_federated.py --use_probabilistic --use_uq_weighting --save_dir abl_uq_yes

# Without UQ weighting (uniform)
python3 train_federated.py --use_probabilistic --save_dir abl_uq_no
```

#### 3. Effect of Different Client Ratios

Edit `data_loader_federated.py`, line 122-124:

```python
# Try different ratios
num_multimodal_clients: int = 3  # instead of 5
num_unimodal_clients: int = 7    # instead of 5
```

Then run experiments to see how performance changes.

### Resume Training from Checkpoint

```python
import torch
from train_federated import main
import argparse

# Load checkpoint
checkpoint = torch.load('checkpoints_prob/checkpoint_round_15.pth')

# Continue training...
# (You'll need to modify train_federated.py to support resume)
```

### Custom Evaluation

```python
from train_federated import evaluate_model
from multimodal_model import get_model
from probabilistic_models import ProbabilisticFIN
from data_loader_federated import create_test_dataloader
import torch

# Load model
model = get_model()
fin_i2t = ProbabilisticFIN()

checkpoint = torch.load('checkpoints_prob/checkpoint_round_30.pth')
model.load_state_dict(checkpoint['model_state'])
fin_i2t.load_state_dict(checkpoint['fin_i2t_state'])

# Evaluate
test_loader = create_test_dataloader(batch_size=32)
metrics = evaluate_model(model, fin_i2t, test_loader, device='cuda')

print(f"AUC: {metrics['auc']:.4f}")
print(f"F1: {metrics['f1']:.4f}")
```

---

## ❓ FAQ

### Q1: How long does training take?

**A:** Depends on hardware:
- **With GPU (e.g., RTX 3090)**: 2-4 hours for 30 rounds
- **With CPU**: 8-12 hours for 30 rounds
- **Quick test (5 rounds)**: 20-40 minutes on GPU, 1-2 hours on CPU

### Q2: What should my final AUC be?

**A:** Based on the paper:
- **Baseline (Original FIN)**: ~0.86
- **Your Method**: ~0.87-0.89 (1-3% improvement)
- **If you get < 0.80**: Something is wrong, check logs

### Q3: Can I use a different dataset?

**A:** Yes! You need to:
1. Organize your data like `Fed_Data` structure
2. Modify `data_loader_federated.py` to load your data
3. Update `num_classes` if not 14 diseases

### Q4: What's the difference between the 4 experiments?

| Experiment | Probabilistic FIN | UQ Weighting | Purpose |
|------------|-------------------|--------------|---------|
| 1. Yours | ✅ Yes | ✅ Yes | **Main method** |
| 2. Baseline | ❌ No | ❌ No | **Compare against** |
| 3. Ablation 1 | ✅ Yes | ❌ No | Show UQ weighting helps |
| 4. Ablation 2 | ❌ No | ✅ Yes | Show Beta-NLL helps |

### Q5: How do I know if variance explosion is happening?

**Signs:**
- Loss becomes NaN
- Loss increases instead of decreasing
- Alpha/beta values > 100 or < 0.1
- Uncertainty values > 0.5

**Check:**
```bash
# Look for these in logs
grep -E "(nan|inf)" checkpoints_prob/log.txt
```

### Q6: Can I run multiple experiments in parallel?

**A:** Yes, but be careful with GPU memory:

```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 python3 train_federated.py --save_dir exp1 &

# Terminal 2
CUDA_VISIBLE_DEVICES=1 python3 train_federated.py --save_dir exp2 &
```

### Q7: What if I only have CPU?

**A:** It works, just slower. The code automatically uses CPU if no GPU is available.

To force CPU:
```python
# Edit train_federated.py, line 311
device = 'cpu'  # Force CPU
```

### Q8: How do I cite this work?

**Original paper:**
```bibtex
@article{poudel2025fedfeatgen,
  title={Multimodal Federated Learning With Missing Modalities
         through Feature Imputation Network},
  author={Poudel, Pranav and others},
  journal={arXiv preprint arXiv:2505.20232},
  year={2025}
}
```

**Your extension:**
- After you publish, add your own citation!

### Q9: Can I modify the architecture?

**A:** Yes! Main things to modify:

1. **Feature dimension**: Edit `multimodal_model.py`, line 13
2. **FIN layers**: Edit `probabilistic_models.py`, line 103-105
3. **Fusion method**: Edit `multimodal_model.py`, line 108-110

### Q10: Where can I get help?

- Check `IMPLEMENTATION_SUMMARY.md` for details
- Read `README_probabilistic.md` for technical info
- Run `python3 test_setup.py` to diagnose issues
- Check the error messages in troubleshooting section above

---

## 📝 Quick Reference Commands

### Essential Commands

```bash
# 1. Test everything works
python3 test_setup.py

# 2. Run your method (quick test)
python3 train_federated.py --use_probabilistic --use_uq_weighting --num_rounds 5 --eval_every 1

# 3. Run your method (full)
python3 train_federated.py --use_probabilistic --use_uq_weighting --num_rounds 30

# 4. Run baseline
python3 train_federated.py --num_rounds 30 --save_dir checkpoints_det

# 5. Visualize results
python3 visualize_results.py

# 6. Interactive menu
./quick_start.sh
```

### Useful Checks

```bash
# Check GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check data exists
ls Fed_Data/splits/ | wc -l  # Should show 10+ files

# Check disk space
df -h .

# Monitor GPU usage (in another terminal)
watch -n 1 nvidia-smi

# Monitor training logs
tail -f checkpoints_prob/log.txt  # If you add logging
```

---

## 🎯 Recommended Workflow

### First Time Setup (30 minutes)

1. Install dependencies
2. Run test script
3. Run quick test (5 rounds)
4. Verify results look reasonable

### Full Experiments (1-2 days)

1. **Day 1 Morning:** Start your method (30 rounds) → 3-4 hours
2. **Day 1 Afternoon:** Start baseline (30 rounds) → 3-4 hours
3. **Day 2:** Run ablations if needed
4. **Day 2 Evening:** Visualize and analyze results

### For Your Paper

1. Run all 4 experiments (yours + baseline + 2 ablations)
2. Generate plots with `visualize_results.py`
3. Create table with AUC/F1 improvements
4. Discuss uncertainty weighting benefits
5. Show sample uncertainty values

---

## ✅ Final Checklist

Before running full experiments:

- [ ] Ran `python3 test_setup.py` successfully
- [ ] Ran quick test (5 rounds) without errors
- [ ] Checked AUC > 0.7 in quick test
- [ ] Verified no NaN/inf in losses
- [ ] Have enough disk space (~10GB)
- [ ] Have time for full run (2-4 hours)

Ready to run!

```bash
python3 train_federated.py --use_probabilistic --use_uq_weighting
```

---

**Good luck with your experiments! 🚀**

For questions or issues, check the troubleshooting section or review the implementation files.
