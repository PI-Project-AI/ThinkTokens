### V18 Cloud Deployment Guide

This document provides instructions for deploying and monitoring the V18 Air-Gap VQ Transformer on a cloud GPU instance (e.g., OVH, Lambda, RunPod).

**Update:** This guide has been updated to reflect the fully self-contained V18 structure.

---

## 1. Prerequisites
*   **Cloud Instance:** An L4 or H100 GPU instance with Ubuntu/Debian Linux.
*   **SSH Access:** Familiarity with `ssh` for connecting to your instance.
*   **Git:** Basic Git knowledge (optional if using SCP).
*   **Python 3.10+ & pip:** Installed on the instance.
*   **CUDA Drivers:** Pre-installed on the instance (standard on most cloud AI images).

## 2. Setup on the Cloud Instance

### 2.1 Copy Files
Since you are deploying from your local machine, copy the `nano_architectures_v18` folder to your instance.

```bash
# On your local machine
scp -r nano_architectures_v18 ubuntu@<YOUR_INSTANCE_IP>:~/projects/TT_v18/
```

### 2.2 Setup Environment
Connect to your instance and set up the Python environment.

```bash
ssh ubuntu@<YOUR_INSTANCE_IP>
cd projects/TT_v18/nano_architectures_v18

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2.3 Verify GPU
Ensure PyTorch sees your GPU.
```bash
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

## 3. Launching Training

The training process is automated by `run_v18.sh`. It handles:
1.  Downloading the TinyStories dataset (train & valid) directly into the `nano_architectures_v18` folder.
2.  Running Phase 1 (Predictive Pre-training).
3.  Running Phase 2 (Reasoning Fine-tune).

It's essential to run this in a detached session (e.g., `screen` or `tmux`) so it continues if your SSH connection drops.

### 3.1 Start a Detached Session
```bash
screen -S v18_training
```

### 3.2 Execute the Training Script
```bash
# Ensure you are in the nano_architectures_v18 directory and venv is active
source .venv/bin/activate
chmod +x run_v18.sh
./run_v18.sh
```

**Crucial Check:**
Watch the first few lines of output. You should see:
*   `Vocab Size: ~5000` (NOT 23!)
*   `Model Parameters: ~190M`

### 3.3 Detach
Press `Ctrl+A` then `D` to detach. The training will continue in the background.

## 4. Monitoring Training

### 4.1 Real-time Logs
Inside your `screen` session, you can monitor output. If you detached, re-attach first:
```bash
screen -r v18_training
```

### 4.2 Check GPU Utilization
Open another terminal/SSH session to monitor GPU usage:
```bash
watch -n 1 nvidia-smi
```

### 4.3 Review Results
Results are saved in:
*   `results_phase1/`
*   `results_phase2/`

## 5. Retrieving Results

Once training is complete, use `scp` to copy results to your local machine.

```bash
# On your local machine
scp -r ubuntu@<YOUR_INSTANCE_IP>:~/projects/TT_v18/nano_architectures_v18/results_phase1 ./results_phase1_v18
scp -r ubuntu@<YOUR_INSTANCE_IP>:~/projects/TT_v18/nano_architectures_v18/results_phase2 ./results_phase2_v18
```

---
Good luck with V18!