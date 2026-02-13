import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')
from utils.perturbations import apply_perturbation

def generate_perturbed_dataset(task, perturbation_type, data_dir, output_file):
    """Generate perturbed query dataset."""
    from datasets import load_dataset

    # Load original queries
    dataset = load_dataset(data_dir, 'examples')[task]

    output_examples = []
    for example in dataset:
        query = example['query']
        
        # Parse perturbation type and parameters
        if perturbation_type.startswith('paraphrase_'):
            variant_id = int(perturbation_type.split('_')[1])
            perturbed = apply_perturbation(query, 'paraphrase', variant_id=variant_id)

        elif perturbation_type.startswith('synonym_'):
            num_replacements = int(perturbation_type.split('_')[1]) + 1
            perturbed = apply_perturbation(query, 'synonym', num_replacements=num_replacements)

        elif perturbation_type.startswith('adversarial_'):
            num_tokens = int(perturbation_type.split('_')[1]) + 1
            perturbed = apply_perturbation(query, 'adversarial', num_tokens=num_tokens)

        elif perturbation_type == 'length_expand':
            perturbed = apply_perturbation(query, 'length_expand')

        elif perturbation_type == 'length_contract':
            perturbed = apply_perturbation(query, 'length_contract')

        else:
            perturbed = query

        # Create updated example
        new_example = dict(example)
        new_example['query'] = perturbed
        output_examples.append(new_example)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(output_examples, f, indent=2)

    print(f"Generated {len(output_examples)} perturbed queries for {task}/{perturbation_type}")
    return output_file


if __name__ == '__main__':
    task = sys.argv[1]
    perturbation = sys.argv[2]
    data_dir = sys.argv[3]
    output_file = sys.argv[4]

    generate_perturbed_dataset(task, perturbation, data_dir, output_file)
