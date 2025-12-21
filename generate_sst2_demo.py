"""
Generate SST-2 demo dataset with perturbations for Collapse Index analysis.
Uses HuggingFace datasets + nlpaug for realistic perturbations.

Output: CSV in the format your CI pipeline expects.
"""

import pandas as pd
from datasets import load_dataset
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from transformers import pipeline
from tqdm import tqdm
import hashlib

# Config
N_SAMPLES = 500  # Number of base examples from SST-2 validation set
VARIANTS_PER_SAMPLE = 3  # How many perturbations per base example
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"  # Public SST-2 model
OUTPUT_PATH = "data/checklist/sst2_ci_demo.csv"

def generate_base_id(text):
    """Generate deterministic base_id from text."""
    return hashlib.md5(text.encode()).hexdigest()[:12]

def create_perturbations(text, n=3):
    """Generate n perturbations of text using various methods."""
    perturbations = []
    
    # Typo augmenter (keyboard distance)
    typo_aug = nac.KeyboardAug(aug_char_max=2, aug_word_p=0.3)
    
    # Synonym augmenter
    synonym_aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.3)
    
    # Contraction augmenter
    contraction_aug = naw.AntonymAug(aug_p=0.2)
    
    methods = [typo_aug, synonym_aug]
    
    for i in range(n):
        try:
            aug = methods[i % len(methods)]
            perturbed = aug.augment(text)
            if isinstance(perturbed, list):
                perturbed = perturbed[0]
            perturbations.append(perturbed)
        except Exception as e:
            # Fallback: simple character swap
            perturbed = text.replace('the', 'teh', 1) if 'the' in text else text + '.'
            perturbations.append(perturbed)
    
    return perturbations

def main():
    print("Loading SST-2 validation set...")
    dataset = load_dataset("glue", "sst2", split="validation")
    
    # Sample N_SAMPLES from validation
    sampled = dataset.shuffle(seed=42).select(range(N_SAMPLES))
    
    print(f"Generating {VARIANTS_PER_SAMPLE} perturbations per example...")
    
    # Prepare data rows
    rows = []
    
    for idx, example in enumerate(tqdm(sampled)):
        text = example['sentence']
        true_label = example['label']  # 0=negative, 1=positive
        base_id = f"sst2_{idx:04d}"
        
        # Base example (variant_id = 'base')
        rows.append({
            'id': base_id,
            'variant_id': 'base',
            'text': text,
            'true_label': 'positive' if true_label == 1 else 'negative',
            'pred_label': None,  # Will fill after model inference
            'confidence': None
        })
        
        # Generate perturbations
        perturbations = create_perturbations(text, VARIANTS_PER_SAMPLE)
        
        for var_idx, perturbed_text in enumerate(perturbations, 1):
            rows.append({
                'id': base_id,
                'variant_id': f'v{var_idx}',
                'text': perturbed_text,
                'true_label': 'positive' if true_label == 1 else 'negative',
                'pred_label': None,
                'confidence': None
            })
    
    print(f"Running model inference on {len(rows)} examples...")
    
    # Load model
    classifier = pipeline("sentiment-analysis", model=MODEL_NAME, device=-1)  # CPU
    
    # Batch inference
    texts = [row['text'] for row in rows]
    predictions = []
    
    batch_size = 32
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        batch_preds = classifier(batch)
        predictions.extend(batch_preds)
    
    # Fill in predictions
    for row, pred in zip(rows, predictions):
        # DistilBERT returns 'LABEL_0' (negative) or 'LABEL_1' (positive)
        label_map = {'LABEL_0': 'negative', 'LABEL_1': 'positive', 'NEGATIVE': 'negative', 'POSITIVE': 'positive'}
        row['pred_label'] = label_map.get(pred['label'].upper(), pred['label'].lower())
        row['confidence'] = pred['score']
    
    # Save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"DONE! Saved to {OUTPUT_PATH}")
    print(f"Stats:")
    print(f"   - Total examples: {len(df)}")
    print(f"   - Base examples: {N_SAMPLES}")
    print(f"   - Variants per base: {VARIANTS_PER_SAMPLE + 1} (including base)")
    print(f"   - Model accuracy (all): {(df['pred_label'] == df['true_label']).mean():.2%}")
    
    # Quick flip analysis
    flip_count = 0
    for case_id in df['id'].unique():
        group = df[df['id'] == case_id]
        predictions = group['pred_label'].unique()
        if len(predictions) > 1:
            flip_count += 1
    
    flip_rate = flip_count / N_SAMPLES
    print(f"   - Flip rate: {flip_rate:.1%} ({flip_count}/{N_SAMPLES} base examples)")
    print(f"\nNext: python -m ci analyze {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
