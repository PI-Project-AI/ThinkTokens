#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.
echo "V17_ter Phase 1..."
.venv/bin/python air_gap/v17_ter/train_phase1.py
echo "V17_ter Phase 2..."
.venv/bin/python air_gap/v17_ter/train_phase2.py
