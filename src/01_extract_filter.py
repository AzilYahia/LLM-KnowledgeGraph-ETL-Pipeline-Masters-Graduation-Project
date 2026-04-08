import pandas as pd
import yaml
import os
import json


def load_config():
    with open('../config.yaml', 'r') as file:
        return yaml.safe_load(file)


def extract_and_filter():
    config = load_config()
    hosp_dir = config['paths']['hosp_dir']
    note_dir = config['paths']['note_dir']
    proc_dir = config['paths']['processed_data']

    os.makedirs(proc_dir, exist_ok=True)

    stats = {}  # To log filtering stats for the thesis

    # ---------------------------------------------------------
    # Phase 2: Cohort Filtering (Inner Join hadm_id)
    # ---------------------------------------------------------
    print("1. Extracting valid hadm_ids from discharge summaries...")
    notes_path = os.path.join(note_dir, config['files']['notes'])
    df_notes = pd.read_csv(notes_path, usecols=['hadm_id'])
    valid_hadm_ids = set(df_notes['hadm_id'].dropna().unique())
    stats['valid_hadm_ids'] = len(valid_hadm_ids)
    print(f"   -> {len(valid_hadm_ids)} unique admissions with a discharge note.")
    del df_notes

    print("\n2. Filtering 'admissions.csv'...")
    df_adm = pd.read_csv(os.path.join(hosp_dir, config['files']['admissions']))
    before = len(df_adm)
    df_adm_filtered = df_adm[df_adm['hadm_id'].isin(valid_hadm_ids)]
    valid_subject_ids = set(df_adm_filtered['subject_id'].unique())
    stats['admissions'] = {'before': before, 'after': len(df_adm_filtered)}
    print(f"   -> {before} -> {len(df_adm_filtered)} admissions | {len(valid_subject_ids)} unique patients.")

    print("\n3. Filtering 'patients.csv'...")
    df_pat = pd.read_csv(os.path.join(hosp_dir, config['files']['patients']))
    before = len(df_pat)
    df_pat_filtered = df_pat[df_pat['subject_id'].isin(valid_subject_ids)]
    stats['patients'] = {'before': before, 'after': len(df_pat_filtered)}
    print(f"   -> {before} -> {len(df_pat_filtered)} patients.")

    print("\n4. Filtering 'diagnoses_icd.csv'...")
    df_diag = pd.read_csv(os.path.join(hosp_dir, config['files']['diagnoses']))
    before = len(df_diag)
    df_diag_filtered = df_diag[df_diag['hadm_id'].isin(valid_hadm_ids)]
    stats['diagnoses_icd'] = {'before': before, 'after': len(df_diag_filtered)}
    print(f"   -> {before} -> {len(df_diag_filtered)} diagnosis records.")

    # ---------------------------------------------------------
    # NEW: procedures_icd
    # ---------------------------------------------------------
    print("\n5. Filtering 'procedures_icd.csv'...")
    procedures_path = os.path.join(hosp_dir, config['files']['procedures'])
    df_proc = pd.read_csv(procedures_path)
    before = len(df_proc)
    df_proc_filtered = df_proc[df_proc['hadm_id'].isin(valid_hadm_ids)]
    stats['procedures_icd'] = {'before': before, 'after': len(df_proc_filtered)}
    print(f"   -> {before} -> {len(df_proc_filtered)} procedure records.")

    # ---------------------------------------------------------
    # NEW: labevents (chunked — very large file)
    # ---------------------------------------------------------
    print("\n6. Filtering 'labevents.csv' (chunked)...")
    labevents_path = os.path.join(hosp_dir, config['files']['labevents'])
    chunk_size = config['pipeline']['chunk_size']
    valid_lab_chunks = []
    total_lab_before = 0

    for chunk in pd.read_csv(labevents_path, chunksize=chunk_size, low_memory=False):
        total_lab_before += len(chunk)
        filtered = chunk[chunk['hadm_id'].isin(valid_hadm_ids)]
        if not filtered.empty:
            valid_lab_chunks.append(filtered)

    df_lab_filtered = pd.concat(valid_lab_chunks, ignore_index=True)
    stats['labevents'] = {'before': total_lab_before, 'after': len(df_lab_filtered)}
    print(f"   -> {total_lab_before} -> {len(df_lab_filtered)} lab event records.")

    # ---------------------------------------------------------
    # Saving outputs
    # ---------------------------------------------------------
    print("\n7. Saving filtered cohort to processed folder...")
    df_adm_filtered.to_csv(os.path.join(proc_dir, 'filtered_admissions.csv'), index=False)
    df_pat_filtered.to_csv(os.path.join(proc_dir, 'filtered_patients.csv'), index=False)
    df_diag_filtered.to_csv(os.path.join(proc_dir, 'filtered_diagnoses_icd.csv'), index=False)
    df_proc_filtered.to_csv(os.path.join(proc_dir, 'filtered_procedures_icd.csv'), index=False)  # NEW
    df_lab_filtered.to_csv(os.path.join(proc_dir, 'filtered_labevents.csv'), index=False)        # NEW

    # Save stats log
    with open(os.path.join(proc_dir, 'stats_phase1_filtering.json'), 'w') as f:
        json.dump(stats, f, indent=4)

    print("\nPhase 1 & 2 Complete! Cohort filtered and saved.")
    print(f"Stats saved to stats_phase1_filtering.json")


if __name__ == "__main__":
    extract_and_filter()