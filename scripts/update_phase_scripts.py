#!/usr/bin/env python3
"""
Script to identify which phase scripts need directory structure updates.
The update needed: change from {task}_{model} to {model}/{task}
"""

import os
from pathlib import Path
import re

def analyze_phase_script(script_path):
    """Analyze a phase script to see if it needs updating."""
    with open(script_path, 'r') as f:
        content = f.read()

    # Look for patterns like:  output_path="${OUTPUT_DIR}/${task}_${model}"
    pattern1 = r'\$\{OUTPUT_DIR\}/\$\{task\}_\$\{model\}'
    pattern2 = r'\$\{OUTPUT_DIR\}/\$\{model\}_\$\{task\}'

    has_task_model = bool(re.search(pattern1, content))
    has_model_task = bool(re.search(pattern2, content))

    return {
        'has_task_model': has_task_model,
        'has_model_task': has_model_task,
        'needs_update': has_task_model or has_model_task
    }

def main():
    scripts_dir = Path(__file__).parent
    phase_scripts = sorted(scripts_dir.glob('phase*.sh'))

    print("="*70)
    print("Phase Scripts Analysis")
    print("="*70)

    needs_update = []
    already_good = []
    special_cases = []

    for script in phase_scripts:
        phase_num = script.stem
        analysis = analyze_phase_script(script)

        # Skip phase3 (corpus scaling - different structure)
        if 'phase3' in script.name:
            special_cases.append(script.name)
            print(f"\n{script.name}: SKIP (special structure - corpus scaling)")
            continue

        if analysis['needs_update']:
            needs_update.append(script.name)
            print(f"\n{script.name}: NEEDS UPDATE")
            print(f"  - Has task_model pattern: {analysis['has_task_model']}")
            print(f"  - Has model_task pattern: {analysis['has_model_task']}")
        else:
            # Check if it's a special phase (quantization, length, etc.)
            with open(script, 'r') as f:
                content = f.read()
            if '${task}_${' in content and '${model}' not in content:
                special_cases.append(script.name)
                print(f"\n{script.name}: SKIP (special structure - parameter sweep)")
            else:
                already_good.append(script.name)
                print(f"\n{script.name}: OK")

    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"\nScripts needing update: {len(needs_update)}")
    for s in needs_update:
        print(f"  - {s}")

    print(f"\nScripts with special structure: {len(special_cases)}")
    for s in special_cases:
        print(f"  - {s}")

    print(f"\nScripts already OK: {len(already_good)}")
    for s in already_good:
        print(f"  - {s}")

if __name__ == '__main__':
    main()
