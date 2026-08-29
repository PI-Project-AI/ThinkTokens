#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.
echo "V16 Phase 1..."
.venv/bin/python air_gap/v16/train_phase1.py
echo "V16 Phase 2..."
.venv/bin/python air_gap/v16/train_phase2.py
