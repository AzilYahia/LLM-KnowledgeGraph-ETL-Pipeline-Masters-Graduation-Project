# import pandas as pd
# import torch
# import os
# import yaml
# from torch_geometric.data import HeteroData
# from sklearn.preprocessing import LabelEncoder, StandardScaler
#
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#
#
# def load_config():
#     with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as file:
#         return yaml.safe_load(file)
#
#
# def build_pyg_graph():
#     config = load_config()
#     proc_dir = os.path.join(PROJECT_ROOT, config['paths']['processed_data'])
#
#     print("1. Loading Cleaned Data...")
#     df_pat = pd.read_csv(os.path.join(proc_dir, 'cleaned_patients.csv'))
#     df_adm = pd.read_csv(os.path.join(proc_dir, 'cleaned_admissions.csv'))
#     df_diag = pd.read_csv(os.path.join(proc_dir, 'cleaned_diagnoses.csv'))
#     df_presc = pd.read_csv(os.path.join(proc_dir, 'cleaned_prescriptions.csv'))
#
#     # Initialize PyTorch Geometric HeteroData object
#     data = HeteroData()
#
#     print("\n2. Creating ID Mappings (Index Compression)...")
#     # PyG requires node IDs to be continuous integers from 0 to N-1
#     pat_encoder = LabelEncoder()
#     df_pat['mapped_id'] = pat_encoder.fit_transform(df_pat['subject_id'])
#
#     adm_encoder = LabelEncoder()
#     df_adm['mapped_id'] = adm_encoder.fit_transform(df_adm['hadm_id'])
#
#     diag_encoder = LabelEncoder()
#     df_diag['mapped_id'] = diag_encoder.fit_transform(df_diag['icd_code'])
#
#     presc_encoder = LabelEncoder()
#     df_presc['mapped_id'] = presc_encoder.fit_transform(df_presc['drug'])
#
#     print("\n3. Building Node Features (X) and Labels (Y)...")
#
#     # --- PATIENT NODES ---
#     # Convert age to tensor, normalize it
#     scaler = StandardScaler()
#     ages = scaler.fit_transform(df_pat[['anchor_age']].fillna(0))
#     data['Patient'].x = torch.tensor(ages, dtype=torch.float)
#
#     # --- ADMISSION NODES (TARGET NODES) ---
#     # Dummy feature for now (1.0) just to give the node substance for message passing
#     data['Admission'].x = torch.ones((len(df_adm), 1), dtype=torch.float)
#
#     # THE LABEL: hospital_expire_flag (1 = Died, 0 = Survived)
#     labels = df_adm['hospital_expire_flag'].fillna(0).astype(int).values
#     data['Admission'].y = torch.tensor(labels, dtype=torch.long)
#
#     # --- DIAGNOSIS & MEDICATION NODES ---
#     data['Diagnosis'].x = torch.ones((len(diag_encoder.classes_), 1), dtype=torch.float)
#     data['Medication'].x = torch.ones((len(presc_encoder.classes_), 1), dtype=torch.float)
#
#     print("\n4. Building Edges (COO Format)...")
#
#     # Edge: (Patient)-[HAS_ADMISSION]->(Admission)
#     # Map raw IDs to our new continuous integer IDs
#     pat_adm_merged = df_adm.merge(df_pat[['subject_id', 'mapped_id']], on='subject_id', suffixes=('_adm', '_pat'))
#     edge_index_pat_adm = torch.tensor([
#         pat_adm_merged['mapped_id_pat'].values,
#         pat_adm_merged['mapped_id_adm'].values
#     ], dtype=torch.long)
#     data['Patient', 'HAS_ADMISSION', 'Admission'].edge_index = edge_index_pat_adm
#
#     # Edge: (Admission)-[DIAGNOSED_WITH]->(Diagnosis)
#     diag_merged = df_diag.merge(df_adm[['hadm_id', 'mapped_id']], on='hadm_id', suffixes=('_diag', '_adm'))
#     edge_index_adm_diag = torch.tensor([
#         diag_merged['mapped_id_adm'].values,
#         diag_merged['mapped_id_diag'].values
#     ], dtype=torch.long)
#     data['Admission', 'DIAGNOSED_WITH', 'Diagnosis'].edge_index = edge_index_adm_diag
#
#     # Edge: (Admission)-[PRESCRIBED]->(Medication)
#     presc_merged = df_presc.merge(df_adm[['hadm_id', 'mapped_id']], on='hadm_id', suffixes=('_presc', '_adm'))
#     edge_index_adm_presc = torch.tensor([
#         presc_merged['mapped_id_adm'].values,
#         presc_merged['mapped_id_presc'].values
#     ], dtype=torch.long)
#     data['Admission', 'PRESCRIBED', 'Medication'].edge_index = edge_index_adm_presc
#
#     print(f"\nGraph Built Successfully!\n{data}")
#
#     print("\n5. Saving to PyTorch (.pt) format...")
#     torch.save(data, os.path.join(proc_dir, 'mimic_graph.pt'))
#     print("Phase 7 Complete! Ready for GNN Training.")
#
#
# if __name__ == "__main__":
#     build_pyg_graph()


# li lfo9 t3 gemini, li lt7t t3 claude


# script 05_export_pyg.py
import pandas as pd
import torch
import os
import yaml
import json
import numpy as np
from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_config():
    with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as file:
        return yaml.safe_load(file)


def build_admission_features(df_adm, df_pat, df_diag, df_presc, df_proc, df_lab, nlp_data):
    """
    Build a rich feature vector for every Admission node.

    Features:
      1.  anchor_age           — patient age
      2.  los_hours            — length of stay in hours
      3.  admission_type_enc   — categorical encoded
      4.  insurance_enc        — categorical encoded
      5.  marital_status_enc   — categorical encoded
      6.  n_diagnoses          — number of ICD diagnosis codes
      7.  n_medications        — number of distinct drugs
      8.  n_procedures         — number of ICD procedure codes       (NEW)
      9.  n_lab_tests          — number of distinct lab tests         (NEW)
      10. mean_lab_value       — mean of all lab valuenum             (NEW)
      11. n_nlp_symptoms       — count of NER-extracted symptoms      (NEW)
      12. n_nlp_diseases       — count of NER-extracted diseases      (NEW)
      13. n_nlp_medications    — count of NER-extracted medications   (NEW)
    """
    print(" -> Engineering admission features (13 features)...")
    df = df_adm.copy()

    # 1. Patient age
    df = df.merge(df_pat[['subject_id', 'anchor_age']], on='subject_id', how='left')

    # 2. LOS (already computed in script 02, but recalculate as safety net)
    if 'los_hours' not in df.columns:
        df['admittime'] = pd.to_datetime(df['admittime'])
        df['dischtime'] = pd.to_datetime(df['dischtime'])
        df['los_hours'] = (df['dischtime'] - df['admittime']).dt.total_seconds() / 3600.0
    df['los_hours'] = df['los_hours'].clip(lower=0).fillna(0)

    # 3-5. Categorical encoding
    for col in ['admission_type', 'insurance', 'marital_status']:
        df[col] = df[col].fillna('UNKNOWN')
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col])

    # 6. Diagnosis count
    diag_counts = df_diag.groupby('hadm_id').size().rename('n_diagnoses')
    df = df.merge(diag_counts, on='hadm_id', how='left')

    # 7. Medication count
    presc_counts = df_presc.groupby('hadm_id')['drug'].nunique().rename('n_medications')
    df = df.merge(presc_counts, on='hadm_id', how='left')

    # 8. Procedure count (NEW)
    proc_counts = df_proc.groupby('hadm_id').size().rename('n_procedures')
    df = df.merge(proc_counts, on='hadm_id', how='left')

    # 9-10. Lab features (NEW)
    lab_agg = df_lab.groupby('hadm_id').agg(
        n_lab_tests=('itemid', 'nunique'),
        mean_lab_value=('valuenum', 'mean')
    ).reset_index()
    df = df.merge(lab_agg, on='hadm_id', how='left')

    # 11-13. NLP features from NER output (NEW)
    nlp_rows = []
    for hadm_id, vals in nlp_data.items():
        nlp_rows.append({
            'hadm_id': int(hadm_id),
            'n_nlp_symptoms':    len(vals.get('symptoms',    [])),
            'n_nlp_diseases':    len(vals.get('diseases',    [])),
            'n_nlp_medications': len(vals.get('medications', [])),
        })
    df_nlp = pd.DataFrame(nlp_rows)
    df = df.merge(df_nlp, on='hadm_id', how='left')

    feature_cols = [
        'anchor_age', 'los_hours',
        'admission_type_enc', 'insurance_enc', 'marital_status_enc',
        'n_diagnoses', 'n_medications', 'n_procedures',
        'n_lab_tests', 'mean_lab_value',
        'n_nlp_symptoms', 'n_nlp_diseases', 'n_nlp_medications',
    ]
    df[feature_cols] = df[feature_cols].fillna(0)

    scaler = StandardScaler()
    features_np = scaler.fit_transform(df[feature_cols].values)

    print(f"    Admission feature matrix: {features_np.shape}")
    print(f"    Features: {feature_cols}")

    return torch.tensor(features_np, dtype=torch.float)


def build_pyg_graph():
    config = load_config()
    proc_dir = config['paths']['processed_data']

    print("1. Loading Cleaned Data...")
    df_pat   = pd.read_csv(os.path.join(proc_dir, 'cleaned_patients.csv'))
    df_adm   = pd.read_csv(os.path.join(proc_dir, 'cleaned_admissions.csv'))
    df_diag  = pd.read_csv(os.path.join(proc_dir, 'cleaned_diagnoses.csv'))
    df_presc = pd.read_csv(os.path.join(proc_dir, 'cleaned_prescriptions.csv'))
    df_proc  = pd.read_csv(os.path.join(proc_dir, 'cleaned_procedures.csv'))   # NEW
    df_lab   = pd.read_csv(os.path.join(proc_dir, 'cleaned_labevents.csv'))    # NEW

    with open(os.path.join(proc_dir, 'nlp_enriched_properties.json'), 'r', encoding='utf-8') as f:
        nlp_data = json.load(f)

    data = HeteroData()

    print("\n2. Creating ID Mappings...")
    pat_encoder  = LabelEncoder(); df_pat['mapped_id']  = pat_encoder.fit_transform(df_pat['subject_id'])
    adm_encoder  = LabelEncoder(); df_adm['mapped_id']  = adm_encoder.fit_transform(df_adm['hadm_id'])
    diag_encoder = LabelEncoder(); df_diag['mapped_id'] = diag_encoder.fit_transform(df_diag['icd_code'])
    presc_encoder= LabelEncoder(); df_presc['mapped_id']= presc_encoder.fit_transform(df_presc['drug'])
    proc_encoder = LabelEncoder(); df_proc['mapped_id'] = proc_encoder.fit_transform(df_proc['icd_code'])  # NEW
    lab_encoder  = LabelEncoder(); df_lab['mapped_id']  = lab_encoder.fit_transform(df_lab['itemid'])      # NEW

    print("\n3. Building Node Features and Labels...")

    # Patient: normalized age
    scaler = StandardScaler()
    ages = scaler.fit_transform(df_pat[['anchor_age']].fillna(0))
    data['Patient'].x = torch.tensor(ages, dtype=torch.float)

    # Admission: 13 clinical features + NLP
    data['Admission'].x = build_admission_features(
        df_adm, df_pat, df_diag, df_presc, df_proc, df_lab, nlp_data
    )

    # Label: mortality
    labels = df_adm['hospital_expire_flag'].fillna(0).astype(int).values
    data['Admission'].y = torch.tensor(labels, dtype=torch.long)

    # Concept nodes: shared identity → learnable embeddings in GNN
    data['Diagnosis'].x  = torch.ones((len(diag_encoder.classes_),  1), dtype=torch.float)
    data['Medication'].x = torch.ones((len(presc_encoder.classes_), 1), dtype=torch.float)
    data['Procedure'].x  = torch.ones((len(proc_encoder.classes_),  1), dtype=torch.float)  # NEW
    data['LabTest'].x    = torch.ones((len(lab_encoder.classes_),   1), dtype=torch.float)  # NEW

    print("\n4. Building Edges (COO Format)...")

    # Patient -> Admission
    pat_adm = df_adm.merge(df_pat[['subject_id', 'mapped_id']], on='subject_id', suffixes=('_adm', '_pat'))
    data['Patient', 'HAS_ADMISSION', 'Admission'].edge_index = torch.tensor([
        pat_adm['mapped_id_pat'].values,
        pat_adm['mapped_id_adm'].values
    ], dtype=torch.long)

    # Admission -> Diagnosis
    diag_m = df_diag.merge(df_adm[['hadm_id', 'mapped_id']], on='hadm_id', suffixes=('_diag', '_adm'))
    data['Admission', 'DIAGNOSED_WITH', 'Diagnosis'].edge_index = torch.tensor([
        diag_m['mapped_id_adm'].values,
        diag_m['mapped_id_diag'].values
    ], dtype=torch.long)

    # Admission -> Medication
    presc_m = df_presc.merge(df_adm[['hadm_id', 'mapped_id']], on='hadm_id', suffixes=('_presc', '_adm'))
    data['Admission', 'PRESCRIBED', 'Medication'].edge_index = torch.tensor([
        presc_m['mapped_id_adm'].values,
        presc_m['mapped_id_presc'].values
    ], dtype=torch.long)

    # Admission -> Procedure (NEW)
    proc_m = df_proc.merge(df_adm[['hadm_id', 'mapped_id']], on='hadm_id', suffixes=('_proc', '_adm'))
    data['Admission', 'HAS_PROCEDURE', 'Procedure'].edge_index = torch.tensor([
        proc_m['mapped_id_adm'].values,
        proc_m['mapped_id_proc'].values
    ], dtype=torch.long)

    # Admission -> LabTest (NEW) — one edge per unique (hadm_id, itemid) pair
    lab_unique = df_lab.drop_duplicates(subset=['hadm_id', 'itemid'])
    lab_m = lab_unique.merge(df_adm[['hadm_id', 'mapped_id']], on='hadm_id', suffixes=('_lab', '_adm'))
    data['Admission', 'HAS_LAB', 'LabTest'].edge_index = torch.tensor([
        lab_m['mapped_id_adm'].values,
        lab_m['mapped_id_lab'].values
    ], dtype=torch.long)

    print(f"\nGraph built:\n{data}")

    # ---------------------------------------------------------
    # Deterministic train/val/test split — saved inside .pt
    # ---------------------------------------------------------
    print("\n5. Creating and saving deterministic train/val/test masks...")
    num_adm = data['Admission'].num_nodes
    g = torch.Generator()
    g.manual_seed(SEED)
    perm = torch.randperm(num_adm, generator=g)

    train_end = int(0.8 * num_adm)
    val_end   = int(0.9 * num_adm)

    train_mask = torch.zeros(num_adm, dtype=torch.bool)
    val_mask   = torch.zeros(num_adm, dtype=torch.bool)
    test_mask  = torch.zeros(num_adm, dtype=torch.bool)

    train_mask[perm[:train_end]]        = True
    val_mask[perm[train_end:val_end]]   = True
    test_mask[perm[val_end:]]           = True

    data['Admission'].train_mask = train_mask
    data['Admission'].val_mask   = val_mask
    data['Admission'].test_mask  = test_mask

    print(f"   Train: {train_mask.sum().item()} | Val: {val_mask.sum().item()} | Test: {test_mask.sum().item()}")

    # Sanity check
    adm_x = data['Admission'].x
    nonzero = adm_x.nonzero(as_tuple=False).shape[0] / adm_x.numel()
    print(f"\n[SANITY CHECK] Admission features non-zero fraction: {nonzero:.4f}")
    print(f"[SANITY CHECK] Labels — 0: {(data['Admission'].y==0).sum().item()} | 1: {(data['Admission'].y==1).sum().item()}")

    print("\n6. Saving graph to .pt...")
    torch.save(data, os.path.join(proc_dir, 'mimic_graph.pt'))
    print("Done! Graph saved to mimic_graph.pt")


if __name__ == "__main__":
    build_pyg_graph()