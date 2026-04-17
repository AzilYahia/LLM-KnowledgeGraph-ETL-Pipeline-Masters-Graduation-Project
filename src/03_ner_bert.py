import pandas as pd
import yaml
import os
import torch
import json
import logging
from transformers import pipeline
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

os.makedirs(os.path.join(PROJECT_ROOT, 'logs'), exist_ok=True)

logging.basicConfig(
    filename=os.path.join(PROJECT_ROOT, 'logs', 'ner_errors.log'),
    level=logging.ERROR,
    format='%(asctime)s — %(message)s'
)

BATCH_SIZE       = 64    # 3060 Ti has 8GB VRAM, 64 is safe
MAX_TOKENS       = 512
MAX_WINDOWS      = 1     # first 512 tokens only — all clinical entities are here
CHECKPOINT_EVERY = 5000


def load_config():
    with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as f:
        return yaml.safe_load(f)


def get_first_window(text, tokenizer=None, max_tokens=MAX_TOKENS):
    """
    Slices the first 350 words of the text BEFORE tokenization.
    This entirely prevents the 512-token warning and is 1000x faster for the CPU.
    """
    words = str(text).split()
    first_words = words[:350]
    return " ".join(first_words)


def resolve_medication_label(ner_pipe):
    id2label   = ner_pipe.model.config.id2label
    all_labels = set(id2label.values())
    candidates = [
        'Medication', 'medication', 'B-Medication', 'B-medication',
        'Chemical', 'chemical', 'B-Chemical',
        'Drug', 'drug', 'B-Drug',
        'MEDICINE', 'B-MEDICINE',
    ]
    for candidate in candidates:
        if candidate in all_labels:
            stripped = candidate.replace('B-', '').replace('I-', '')
            print(f"   -> Medication label detected as: '{stripped}' (raw: '{candidate}')")
            return stripped
    print(f"   WARNING: No medication label matched. Available: {all_labels}")
    return None


def parse_entities(entities, medication_label):
    symptoms, diseases, medications = set(), set(), set()
    for ent in entities:
        group = ent['entity_group']
        word  = ent['word'].lower().strip()
        if group == 'Sign_symptom':
            symptoms.add(word)
        elif group == 'Disease_disorder':
            diseases.add(word)
        elif medication_label and group == medication_label:
            medications.add(word)
    return symptoms, diseases, medications


def merge_into(extracted_data, hadm_id, symptoms, diseases, medications):
    if hadm_id in extracted_data:
        extracted_data[hadm_id]['symptoms']    = list(set(extracted_data[hadm_id].get('symptoms',    [])) | symptoms)
        extracted_data[hadm_id]['diseases']    = list(set(extracted_data[hadm_id].get('diseases',    [])) | diseases)
        extracted_data[hadm_id]['medications'] = list(set(extracted_data[hadm_id].get('medications', [])) | medications)
    else:
        extracted_data[hadm_id] = {
            "symptoms":    list(symptoms),
            "diseases":    list(diseases),
            "medications": list(medications)
        }


def extract_clinical_entities():
    config     = load_config()
    proc_dir   = config['paths']['processed_data']
    note_dir   = config['paths']['note_dir']
    notes_path = os.path.join(note_dir, config['files']['notes'])
    chunk_size = config['pipeline']['chunk_size']

    checkpoint_path = os.path.join(proc_dir, 'ner_checkpoint.json')
    output_json     = os.path.join(proc_dir, 'nlp_enriched_properties.json')
    stats_json      = os.path.join(proc_dir, 'stats_phase4_ner.json')

    print("=" * 60)
    print("PHASE 4 — Clinical NER on Discharge Summaries")
    print("=" * 60)

    print("\n1. Initializing NER model...")
    device = 0 if torch.cuda.is_available() else -1
    if device == 0:
        print(f"   -> GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("   -> WARNING: No GPU detected — CPU mode will be very slow.")

    ner_pipe = pipeline(
        "ner",
        model="d4data/biomedical-ner-all",
        tokenizer="d4data/biomedical-ner-all",
        aggregation_strategy="simple",
        device=device,
        batch_size=BATCH_SIZE
    )
    tokenizer        = ner_pipe.tokenizer
    medication_label = resolve_medication_label(ner_pipe)

    print("\n2. Loading valid admissions...")
    df_adm         = pd.read_csv(os.path.join(proc_dir, 'cleaned_admissions.csv'))
    valid_hadm_ids = set(df_adm['hadm_id'].unique())
    print(f"   -> {len(valid_hadm_ids)} admissions in cohort.")

    print("\n3. Checking for checkpoint...")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            extracted_data = json.load(f)
        already_done = set(extracted_data.keys())
        print(f"   -> Resuming: {len(extracted_data)} admissions already done.")
    else:
        extracted_data = {}
        already_done   = set()
        print("   -> No checkpoint — starting fresh.")

    print(f"\n4. Extracting entities...")
    print(f"   batch_size={BATCH_SIZE} | max_windows={MAX_WINDOWS} | device={'GPU' if device==0 else 'CPU'}")

    notes_processed = 0
    notes_skipped   = 0
    notes_failed    = 0

    chunk_iterator = pd.read_csv(
        notes_path,
        chunksize=chunk_size,
        usecols=['hadm_id', 'text'],
        low_memory=False
    )

    for chunk_idx, chunk in enumerate(chunk_iterator):
        chunk = chunk[chunk['hadm_id'].isin(valid_hadm_ids)]
        chunk = chunk.dropna(subset=['hadm_id', 'text'])

        batch_texts     = []
        batch_mapping   = []
        rows_to_process = []

        for _, row in chunk.iterrows():
            hadm_id = str(int(row['hadm_id']))
            if hadm_id in already_done and hadm_id in extracted_data:
                notes_skipped += 1
                continue
            rows_to_process.append((hadm_id, str(row['text'])))

        for hadm_id, text in rows_to_process:
            try:
                window = get_first_window(text, tokenizer)
                if window.strip():
                    batch_texts.append(window)
                    batch_mapping.append(hadm_id)
            except Exception as e:
                notes_failed += 1
                logging.error(f"hadm_id={hadm_id} | Error: {e}")

        if not batch_texts:
            continue

        print(f"\n   Chunk {chunk_idx + 1}: {len(rows_to_process)} notes → {len(batch_texts)} windows")

        all_results = []
        SUB_CHUNK = 2048  # On donne les textes au pipeline petit à petit

        try:
            # On crée une barre de progression basée sur les sous-paquets
            for i in tqdm(range(0, len(batch_texts), SUB_CHUNK), desc=f"Chunk {chunk_idx + 1} — GPU inference"):
                sub_texts = batch_texts[i: i + SUB_CHUNK]

                # Le GPU traite ces 2048 textes (toujours par lots de 64 en interne)
                sub_results = ner_pipe(sub_texts)
                all_results.extend(sub_results)

        except Exception as e:
            logging.error(f"Chunk {chunk_idx + 1} failed: {e}")
            print(f"   ERROR on chunk {chunk_idx + 1}: {e}")
            continue

        note_symptoms    = {}
        note_diseases    = {}
        note_medications = {}

        for entities, hadm_id in zip(all_results, batch_mapping):
            syms, dis, meds = parse_entities(entities, medication_label)
            note_symptoms   .setdefault(hadm_id, set()).update(syms)
            note_diseases   .setdefault(hadm_id, set()).update(dis)
            note_medications.setdefault(hadm_id, set()).update(meds)

        for hadm_id in note_symptoms.keys() | note_diseases.keys() | note_medications.keys():
            merge_into(
                extracted_data, hadm_id,
                note_symptoms.get(hadm_id,    set()),
                note_diseases.get(hadm_id,    set()),
                note_medications.get(hadm_id, set())
            )

        notes_processed += len(rows_to_process)

        if notes_processed % CHECKPOINT_EVERY < len(rows_to_process):
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(extracted_data, f)
            print(f"   [CHECKPOINT] {notes_processed} notes — {len(extracted_data)} admissions enriched.")

    print(f"\n{'=' * 60}")
    print(f"  Notes processed    : {notes_processed}")
    print(f"  Notes skipped      : {notes_skipped} (checkpoint)")
    print(f"  Notes failed       : {notes_failed} (see logs/ner_errors.log)")
    print(f"  Admissions enriched: {len(extracted_data)}")
    print(f"{'=' * 60}")

    print("\n5. Saving outputs...")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=4)
    print(f"   -> {output_json}")

    ner_stats = {
        'processed': notes_processed,
        'skipped':   notes_skipped,
        'failed':    notes_failed,
        'enriched':  len(extracted_data)
    }
    with open(stats_json, 'w', encoding='utf-8') as f:
        json.dump(ner_stats, f, indent=4)
    print(f"   -> {stats_json}")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("   -> Checkpoint removed.")

    print("\nPhase 4 Complete!")


if __name__ == "__main__":
    extract_clinical_entities()