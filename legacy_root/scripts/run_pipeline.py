#!/usr/bin/env python3
"""
Master pipeline script to run the complete IR reasoning experiment.
Coordinates: baseline → training → evaluation → analysis
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

def run_command(cmd, description):
    """Run a command and track execution."""
    print(f"\n{'='*80}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with exit code {e.returncode}")
        return False

def main():
    """Main pipeline execution."""
    print(f"\n{'='*80}")
    print("INTERMEDIATE REASONING LANGUAGE - FULL PIPELINE")
    print(f"{'='*80}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Ensure results directory exists
    Path("results").mkdir(exist_ok=True)

    # Track pipeline progress
    pipeline_log = {
        'start_time': datetime.now().isoformat(),
        'steps': []
    }

    steps = [
        (
            ['python', 'eval_baseline.py'],
            'Step 1: Establish Baseline Performance'
        ),
        (
            ['python', 'train_vq.py'],
            'Step 2: Train VQ Model'
        ),
        (
            ['python', 'eval_vq.py'],
            'Step 3: Evaluate VQ Model'
        ),
        (
            ['python', 'analyze_results.py'],
            'Step 4: Analyze Results & Generate Report'
        )
    ]

    # Execute pipeline
    for i, (cmd, description) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] Running: {description}")
        success = run_command(cmd, description)

        if not success:
            print(f"\n✗ Pipeline failed at step {i}")
            pipeline_log['steps'].append({
                'step': i,
                'description': description,
                'status': 'FAILED',
                'timestamp': datetime.now().isoformat()
            })
            break

        pipeline_log['steps'].append({
            'step': i,
            'description': description,
            'status': 'COMPLETED',
            'timestamp': datetime.now().isoformat()
        })

    # Save pipeline log
    pipeline_log['end_time'] = datetime.now().isoformat()
    log_path = Path("results") / "pipeline.json"
    with open(log_path, 'w') as f:
        json.dump(pipeline_log, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print("PIPELINE SUMMARY")
    print(f"{'='*80}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Completed steps: {sum(1 for s in pipeline_log['steps'] if s['status'] == 'COMPLETED')}/{len(steps)}")

    if all(s['status'] == 'COMPLETED' for s in pipeline_log['steps']):
        print("\n✓ All pipeline steps completed successfully!")
        print("\nGenerated artifacts:")
        results_dir = Path("results")
        for file in sorted(results_dir.glob("*")):
            if file.is_file():
                print(f"  • {file.name}")
        print(f"\nResults saved in: {results_dir.absolute()}")
    else:
        print("\n✗ Pipeline did not complete successfully")
        sys.exit(1)

if __name__ == '__main__':
    main()
