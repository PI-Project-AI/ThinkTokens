#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.
echo "Starting Phase 1 (Auto-Encoder)..."
.venv/bin/python air_gap/v15/train_phase1.py
echo "Starting Phase 2 (Reasoning)..."
.venv/bin/python air_gap/v15/train_phase2.py
