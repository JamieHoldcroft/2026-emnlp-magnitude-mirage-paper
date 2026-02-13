"""
Query perturbation generators for robustness evaluation.

Implements various perturbation types from recent robustness research:
- Query paraphrasing
- Synonym replacement
- Adversarial token insertion
- Query length perturbation
"""

import random
import nltk
from nltk.corpus import wordnet
import re

# Download required NLTK data (run once)
try:
    wordnet.synsets('test')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt', quiet=True)


def get_synonyms(word, pos=None):
    """Get synonyms for a word using WordNet."""
    synonyms = set()

    for syn in wordnet.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym.lower() != word.lower():
                synonyms.add(synonym)

    return list(synonyms)


def synonym_replacement(query, num_replacements=1):
    """
    Replace random words with synonyms.

    Args:
        query: Input query string
        num_replacements: Number of words to replace

    Returns:
        Perturbed query string
    """
    words = query.split()

    if len(words) == 0:
        return query

    # Filter out stopwords and very short words
    stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been'])

    replaceable_indices = [i for i, w in enumerate(words)
                           if w.lower() not in stopwords and len(w) > 3]

    if not replaceable_indices:
        return query

    # Randomly select words to replace
    num_to_replace = min(num_replacements, len(replaceable_indices))
    indices_to_replace = random.sample(replaceable_indices, num_to_replace)

    perturbed_words = words.copy()

    for idx in indices_to_replace:
        word = words[idx]
        synonyms = get_synonyms(word)

        if synonyms:
            # Randomly choose a synonym
            replacement = random.choice(synonyms)
            perturbed_words[idx] = replacement

    return ' '.join(perturbed_words)


def paraphrase_query(query, variant_id=0):
    """
    Generate paraphrased variants of the query.

    Args:
        query: Input query string
        variant_id: Which paraphrase variant to generate (0-4)

    Returns:
        Paraphrased query string
    """
    # Template-based paraphrasing strategies
    paraphrase_templates = [
        # Variant 0: Add "I want to know about"
        lambda q: f"I want to know about {q.lower()}",

        # Variant 1: Rephrase as question
        lambda q: f"Can you tell me about {q.lower()}?" if not q.endswith('?') else q,

        # Variant 2: Add "Find information on"
        lambda q: f"Find information on {q.lower()}",

        # Variant 3: Add "What is"
        lambda q: f"What is {q.lower()}?" if not q.lower().startswith('what') else q,

        # Variant 4: Add "Search for details about"
        lambda q: f"Search for details about {q.lower()}",
    ]

    variant_id = variant_id % len(paraphrase_templates)
    return paraphrase_templates[variant_id](query)


def adversarial_insertion(query, num_tokens=1):
    """
    Insert adversarial tokens to test robustness.

    Args:
        query: Input query string
        num_tokens: Number of adversarial tokens to insert

    Returns:
        Perturbed query with adversarial tokens
    """
    # Common adversarial tokens from robustness research
    adversarial_tokens = [
        "IMPORTANT",
        "URGENT",
        "RELEVANT",
        "KEY",
        "CRITICAL",
        "ESSENTIAL",
        "VITAL",
        "[NOISE]",
        "***",
        "!!!",
    ]

    words = query.split()

    for _ in range(num_tokens):
        # Insert at random position
        insert_pos = random.randint(0, len(words))
        adv_token = random.choice(adversarial_tokens)
        words.insert(insert_pos, adv_token)

    return ' '.join(words)


def length_perturbation(query, mode='expand'):
    """
    Perturb query length by adding/removing words.

    Args:
        query: Input query string
        mode: 'expand' or 'contract'

    Returns:
        Length-perturbed query
    """
    words = query.split()

    if mode == 'expand':
        # Add redundant phrases
        expansion_phrases = [
            "related to",
            "information about",
            "details on",
            "concerning",
            "regarding",
        ]

        phrase = random.choice(expansion_phrases)
        return f"{phrase} {query}"

    elif mode == 'contract':
        # Remove non-essential words (stopwords)
        stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                        'to', 'for', 'of', 'with', 'by', 'from'])

        filtered_words = [w for w in words if w.lower() not in stopwords]

        if len(filtered_words) < 2:  # Don't over-contract
            return query

        return ' '.join(filtered_words)

    return query


def apply_perturbation(query, perturbation_type, **kwargs):
    """
    Apply a specific perturbation type to a query.

    Args:
        query: Input query string
        perturbation_type: One of ['paraphrase', 'synonym', 'adversarial', 'length']
        **kwargs: Additional arguments for specific perturbation types

    Returns:
        Perturbed query string
    """
    if perturbation_type == 'paraphrase':
        variant_id = kwargs.get('variant_id', 0)
        return paraphrase_query(query, variant_id)

    elif perturbation_type == 'synonym':
        num_replacements = kwargs.get('num_replacements', 1)
        return synonym_replacement(query, num_replacements)

    elif perturbation_type == 'adversarial':
        num_tokens = kwargs.get('num_tokens', 1)
        return adversarial_insertion(query, num_tokens)

    elif perturbation_type == 'length_expand':
        return length_perturbation(query, mode='expand')

    elif perturbation_type == 'length_contract':
        return length_perturbation(query, mode='contract')

    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}")


def generate_perturbation_variants(query, perturbation_types=None, num_variants_per_type=3):
    """
    Generate multiple perturbation variants of a query.

    Args:
        query: Input query string
        perturbation_types: List of perturbation types to apply
        num_variants_per_type: Number of variants per perturbation type

    Returns:
        Dict mapping perturbation_id -> perturbed_query
    """
    if perturbation_types is None:
        perturbation_types = ['paraphrase', 'synonym', 'adversarial', 'length_expand']

    variants = {'original': query}

    for perturb_type in perturbation_types:
        if perturb_type == 'paraphrase':
            for i in range(min(num_variants_per_type, 5)):  # Max 5 paraphrase templates
                key = f"paraphrase_{i}"
                variants[key] = apply_perturbation(query, 'paraphrase', variant_id=i)

        elif perturb_type == 'synonym':
            for i in range(num_variants_per_type):
                key = f"synonym_{i}"
                # Try different numbers of replacements
                num_replacements = min(i + 1, 3)
                variants[key] = apply_perturbation(query, 'synonym', num_replacements=num_replacements)

        elif perturb_type == 'adversarial':
            for i in range(num_variants_per_type):
                key = f"adversarial_{i}"
                variants[key] = apply_perturbation(query, 'adversarial', num_tokens=i + 1)

        elif perturb_type == 'length_expand':
            key = "length_expand"
            variants[key] = apply_perturbation(query, 'length_expand')

        elif perturb_type == 'length_contract':
            key = "length_contract"
            variants[key] = apply_perturbation(query, 'length_contract')

    return variants
