import pandas as pd
import yaml
import os
import torch
import json
import logging
from transformers import pipeline
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(
    filename=os.path.join(PROJECT_ROOT, 'logs', 'ner_errors.log'),
    level=logging.ERROR,
    format='%(asctime)s — %(message)s'
)


def load_config():
    config_path = os.path.join(PROJECT_ROOT, 'config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def sliding_window(text, tokenizer, max_tokens=512, stride=256):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_text = tokenizer.decode(tokens[start:end], skip_special_tokens=True)
        chunks.append(chunk_text)
        if end == len(tokens):
            break
        start += stride
    return chunks


def resolve_medication_label(ner_pipe):
    """
    The medication label varies across model versions.
    Dynamically detect the correct label name instead of hardcoding it.
    Possible values: 'Medication', 'Chemical', 'MEDICINE', etc.
    """
    id2label = ner_pipe.model.config.id2label
    all_labels = set(id2label.values())

    candidates = ['Medication', 'medication', 'Chemical', 'chemical', 'MEDICINE', 'medicine', 'Drug', 'drug']
    for candidate in candidates:
        if candidate in all_labels:
            print(f"   -> Medication label detected as: '{candidate}'")
            return candidate

    # Fallback: print all labels so the user can add the right one manually
    print(f"   WARNING: No medication label found. Available labels: {all_labels}")
    return None


def extract_clinical_entities():
    config = load_config()

    proc_dir = os.path.join(PROJECT_ROOT, config['paths']['processed_data'])
    note_dir = config['paths']['note_dir']
    notes_path = os.path.join(note_dir, config['files']['notes'])
    chunk_size = config['pipeline']['chunk_size']

    os.makedirs(os.path.join(PROJECT_ROOT, 'logs'), exist_ok=True)

    print("1. Initializing NER model (checking for GPU)...")
    device = 0 if torch.cuda.is_available() else -1
    print(f"   -> {'GPU: ' + torch.cuda.get_device_name(0) if device == 0 else 'WARNING: CPU mode (slow).'}")

    ner_pipe = pipeline(
        "ner",
        model="d4data/biomedical-ner-all",
        tokenizer="d4data/biomedical-ner-all",
        aggregation_strategy="simple",
        device=device
    )
    tokenizer = ner_pipe.tokenizer

    # Dynamically resolve medication label instead of hardcoding
    medication_label = resolve_medication_label(ner_pipe)

    print("\n2. Loading valid admissions...")
    df_adm = pd.read_csv(os.path.join(proc_dir, 'cleaned_admissions.csv'))
    valid_hadm_ids = set(df_adm['hadm_id'].unique())

    print("\n3. Extracting clinical entities (Symptoms / Diseases / Medications)...")
    extracted_data = {}
    notes_processed = 0
    notes_failed = 0

    chunk_iterator = pd.read_csv(notes_path, chunksize=chunk_size, usecols=['hadm_id', 'text'])

    for chunk in chunk_iterator:
        chunk = chunk[chunk['hadm_id'].isin(valid_hadm_ids)]

        for _, row in tqdm(chunk.iterrows(), total=len(chunk), desc="NER Processing"):
            hadm_id = str(int(row['hadm_id']))
            text = str(row['text'])

            try:
                windows = sliding_window(text, tokenizer, max_tokens=512, stride=256)

                symptoms    = set()
                diseases    = set()
                medications = set()

                for window_text in windows:
                    entities = ner_pipe(window_text)
                    for ent in entities:
                        group = ent['entity_group']
                        word  = ent['word'].lower().strip()
                        if group == 'Sign_symptom':
                            symptoms.add(word)
                        elif group == 'Disease_disorder':
                            diseases.add(word)
                        elif medication_label and group == medication_label:
                            medications.add(word)

                if symptoms or diseases or medications:
                    # -------------------------------------------------
                    # FIX: MERGE instead of overwrite if hadm_id already
                    # exists (a patient can have multiple discharge notes)
                    # -------------------------------------------------
                    if hadm_id in extracted_data:
                        extracted_data[hadm_id]['symptoms']    = list(
                            set(extracted_data[hadm_id]['symptoms'])    | symptoms)
                        extracted_data[hadm_id]['diseases']    = list(
                            set(extracted_data[hadm_id]['diseases'])    | diseases)
                        extracted_data[hadm_id]['medications'] = list(
                            set(extracted_data[hadm_id]['medications']) | medications)
                    else:
                        extracted_data[hadm_id] = {
                            "symptoms":    list(symptoms),
                            "diseases":    list(diseases),
                            "medications": list(medications)
                        }

                notes_processed += 1

            except Exception as e:
                notes_failed += 1
                logging.error(f"hadm_id={hadm_id} | Error: {e}")
                continue

    print(f"\n   -> Notes processed: {notes_processed}")
    print(f"   -> Notes failed (logged): {notes_failed}")

    print("\n4. Saving NLP metadata...")
    output_path = os.path.join(proc_dir, 'nlp_enriched_properties.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=4)

    ner_stats = {'processed': notes_processed, 'failed': notes_failed, 'enriched': len(extracted_data)}
    with open(os.path.join(proc_dir, 'stats_phase4_ner.json'), 'w') as f:
        json.dump(ner_stats, f, indent=4)

    print(f"\nPhase 4 Complete! NLP data saved to: {output_path}")


if __name__ == "__main__":
    extract_clinical_entities()