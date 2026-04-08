import pandas as pd
import yaml
import os
import json
from tqdm import tqdm


def load_config():
    with open('../config.yaml', 'r') as file:
        return yaml.safe_load(file)


def clean_tabular_data():
    config = load_config()
    proc_dir = config['paths']['processed_data']
    hosp_dir = config['paths']['hosp_dir']
    chunk_size = config['pipeline']['chunk_size']

    stats = {}

    print("1. Loading the filtered cohort...")
    df_adm = pd.read_csv(os.path.join(proc_dir, 'filtered_admissions.csv'))
    df_pat = pd.read_csv(os.path.join(proc_dir, 'filtered_patients.csv'))
    df_diag = pd.read_csv(os.path.join(proc_dir, 'filtered_diagnoses_icd.csv'))

    valid_hadm_ids = set(df_adm['hadm_id'].unique())
    valid_subject_ids = set(df_adm['subject_id'].unique())

    print("\n2. Cleaning Admissions...")
    initial_adm = len(df_adm)
    df_adm = df_adm.drop_duplicates(subset=['hadm_id'])
    df_adm['admittime'] = pd.to_datetime(df_adm['admittime'])
    df_adm['dischtime'] = pd.to_datetime(df_adm['dischtime'])
    df_adm = df_adm[df_adm['dischtime'] >= df_adm['admittime']]
    # Compute Length of Stay in hours (useful for GNN features + RAG)
    df_adm['los_hours'] = (df_adm['dischtime'] - df_adm['admittime']).dt.total_seconds() / 3600
    stats['admissions'] = {'before': initial_adm, 'after': len(df_adm), 'dropped': initial_adm - len(df_adm)}
    print(f"   -> Dropped {initial_adm - len(df_adm)} invalid/duplicate rows. LOS computed.")

    print("\n3. Cleaning Patients (filter underage)...")
    initial_pat = len(df_pat)
    df_pat = df_pat[df_pat['subject_id'].isin(valid_subject_ids)]
    # Exclude pediatric patients (anchor_age < 18)
    df_pat = df_pat[df_pat['anchor_age'] >= 18]
    stats['patients'] = {'before': initial_pat, 'after': len(df_pat), 'dropped': initial_pat - len(df_pat)}
    print(f"   -> Dropped {initial_pat - len(df_pat)} rows (underage or outside cohort).")

    print("\n4. Cleaning Diagnoses...")
    initial_diag = len(df_diag)
    df_diag = df_diag.dropna(subset=['icd_code'])
    df_diag = df_diag.drop_duplicates()
    stats['diagnoses'] = {'before': initial_diag, 'after': len(df_diag), 'dropped': initial_diag - len(df_diag)}
    print(f"   -> Dropped {initial_diag - len(df_diag)} null/duplicate rows.")

    # ---------------------------------------------------------
    # NEW: Clean Procedures
    # ---------------------------------------------------------
    print("\n5. Cleaning Procedures...")
    df_proc = pd.read_csv(os.path.join(proc_dir, 'filtered_procedures_icd.csv'))
    initial_proc = len(df_proc)
    df_proc = df_proc.dropna(subset=['icd_code'])
    df_proc = df_proc.drop_duplicates()
    stats['procedures'] = {'before': initial_proc, 'after': len(df_proc), 'dropped': initial_proc - len(df_proc)}
    print(f"   -> Dropped {initial_proc - len(df_proc)} null/duplicate rows.")

    # ---------------------------------------------------------
    # NEW: Clean Labevents (physiological range filtering)
    # ---------------------------------------------------------
    print("\n6. Cleaning Labevents (chunked + physiological range filter)...")
    labevents_path = os.path.join(proc_dir, 'filtered_labevents.csv')
    valid_lab_chunks = []
    total_before = 0
    total_after = 0

    # Physiological plausibility bounds (catch obvious sensor errors)
    VALUE_MIN = -1000
    VALUE_MAX = 1000000

    for chunk in tqdm(pd.read_csv(labevents_path, chunksize=chunk_size, low_memory=False), desc="Cleaning Labevents"):
        total_before += len(chunk)
        # Drop rows with no result value
        chunk = chunk.dropna(subset=['valuenum'])
        # Drop physiologically implausible values
        chunk = chunk[(chunk['valuenum'] >= VALUE_MIN) & (chunk['valuenum'] <= VALUE_MAX)]
        # Drop duplicate measurements (same admission, same test, same time)
        chunk = chunk.drop_duplicates(subset=['hadm_id', 'itemid', 'charttime'])
        total_after += len(chunk)
        valid_lab_chunks.append(chunk)

    df_lab = pd.concat(valid_lab_chunks, ignore_index=True)
    stats['labevents'] = {'before': total_before, 'after': total_after, 'dropped': total_before - total_after}
    print(f"   -> {total_before} -> {total_after} lab records kept.")

    # ---------------------------------------------------------
    # Extract & Clean Prescriptions (chunked)
    # ---------------------------------------------------------
    print("\n7. Extracting & Cleaning Prescriptions (chunked)...")
    prescriptions_path = os.path.join(hosp_dir, 'prescriptions.csv')
    valid_presc = []
    total_presc_before = 0

    for chunk in tqdm(pd.read_csv(prescriptions_path, chunksize=chunk_size, low_memory=False), desc="Parsing Prescriptions"):
        total_presc_before += len(chunk)
        chunk = chunk[chunk['hadm_id'].isin(valid_hadm_ids)]
        chunk = chunk.dropna(subset=['drug'])
        valid_presc.append(chunk)

    df_presc = pd.concat(valid_presc, ignore_index=True)
    initial_presc = len(df_presc)
    df_presc = df_presc.drop_duplicates(subset=['hadm_id', 'drug', 'starttime'])
    stats['prescriptions'] = {'before': total_presc_before, 'after': len(df_presc), 'dropped': initial_presc - len(df_presc)}
    print(f"   -> {len(df_presc)} valid prescription records kept.")

    # ---------------------------------------------------------
    # Saving
    # ---------------------------------------------------------
    print("\n8. Saving fully cleaned data...")
    df_adm.to_csv(os.path.join(proc_dir, 'cleaned_admissions.csv'), index=False)
    df_pat.to_csv(os.path.join(proc_dir, 'cleaned_patients.csv'), index=False)
    df_diag.to_csv(os.path.join(proc_dir, 'cleaned_diagnoses.csv'), index=False)
    df_presc.to_csv(os.path.join(proc_dir, 'cleaned_prescriptions.csv'), index=False)
    df_proc.to_csv(os.path.join(proc_dir, 'cleaned_procedures.csv'), index=False)  # NEW
    df_lab.to_csv(os.path.join(proc_dir, 'cleaned_labevents.csv'), index=False)    # NEW

    with open(os.path.join(proc_dir, 'stats_phase3_cleaning.json'), 'w') as f:
        json.dump(stats, f, indent=4)

    print("\nPhase 3 Complete! All tables cleaned and saved.")


if __name__ == "__main__":
    clean_tabular_data()