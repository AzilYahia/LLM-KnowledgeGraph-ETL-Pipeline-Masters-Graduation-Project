"""
07_graph_rag.py — Graph RAG + Llama-3 Justification Module (Phase 10)

Workflow per patient query:
  1. Load pre-computed admission embeddings (from script 06)
  2. FAISS: find the K most similar admissions
  3. Neo4j: retrieve their clinical subgraphs
  4. Build a structured natural-language context
  5. Call Llama-3 (via Groq API) with context + GNN prediction
  6. Return and save a human-readable clinical justification
"""

import os
import json
import time

import torch
import faiss
import numpy as np
import pandas as pd
import yaml
from neo4j import GraphDatabase
from dotenv import load_dotenv
from groq import Groq

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def load_config():
    with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as file:
        return yaml.safe_load(file)


# ---------------------------------------------------------
# 1. FAISS — Similar Patient Retrieval
# ---------------------------------------------------------
class FAISSRetriever:
    def __init__(self, embeddings: torch.Tensor):
        self.embeddings_np = embeddings.numpy().astype('float32')
        D = self.embeddings_np.shape[1]
        self.index = faiss.IndexFlatL2(D)
        self.index.add(self.embeddings_np)
        print(f"   FAISS index built: {self.index.ntotal} admissions, dim={D}")

    def find_similar(self, query_idx: int, k: int = 5):
        query_vec = self.embeddings_np[query_idx:query_idx + 1]
        distances, indices = self.index.search(query_vec, k + 1)
        mask = indices[0] != query_idx
        return distances[0][mask][:k], indices[0][mask][:k]


# ---------------------------------------------------------
# 2. Neo4j Subgraph Retriever
# ---------------------------------------------------------
class Neo4jSubgraphRetriever:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_admission_subgraph(self, hadm_id: int) -> dict:
        query = """
        MATCH (a:Admission {id: $hadm_id})
        OPTIONAL MATCH (a)-[:DIAGNOSED_WITH]->(d:Diagnosis)
        OPTIONAL MATCH (a)-[:PRESCRIBED]->(m:Medication)
        OPTIONAL MATCH (a)-[:HAS_PROCEDURE]->(pr:Procedure)
        OPTIONAL MATCH (a)-[lr:HAS_LAB]->(l:LabTest)
        RETURN
            a.admission_type        AS admission_type,
            a.insurance             AS insurance,
            a.los_hours             AS los_hours,
            a.hospital_expire_flag  AS died,
            a.extracted_symptoms    AS symptoms,
            a.extracted_diseases    AS nlp_diseases,
            a.extracted_medications AS nlp_medications,
            collect(DISTINCT d.code)                              AS diagnoses,
            collect(DISTINCT m.name)                              AS medications,
            collect(DISTINCT pr.code)                             AS procedures,
            collect(DISTINCT {item: l.item_id, mean: lr.mean_value}) AS lab_results
        """
        with self.driver.session() as session:
            result = session.run(query, hadm_id=hadm_id).single()
            if result is None:
                return {}
            return dict(result)


# ---------------------------------------------------------
# 3. Context Builder
# ---------------------------------------------------------
def build_rag_context(similar_subgraphs: list, query_subgraph: dict) -> str:
    lines = []

    los = query_subgraph.get('los_hours')
    los_str = f"{los:.1f} hours" if isinstance(los, (int, float)) else "N/A"

    lines.append("=== PATIENT QUERY ===")
    lines.append(f"Admission type : {query_subgraph.get('admission_type', 'N/A')}")
    lines.append(f"Length of stay : {los_str}")
    lines.append(f"Diagnoses      : {', '.join(query_subgraph.get('diagnoses', [])) or 'None'}")
    lines.append(f"Medications    : {', '.join(query_subgraph.get('medications', [])) or 'None'}")
    lines.append(f"Procedures     : {', '.join(query_subgraph.get('procedures', [])) or 'None'}")
    lines.append(f"NLP Symptoms   : {', '.join(query_subgraph.get('symptoms', []) or []) or 'None'}")
    lines.append(f"NLP Diseases   : {', '.join(query_subgraph.get('nlp_diseases', []) or []) or 'None'}")
    lines.append("")

    lines.append("=== SIMILAR PAST PATIENTS (retrieved by GNN embeddings) ===")
    for i, sg in enumerate(similar_subgraphs, 1):
        outcome = "DIED" if sg.get('died') == 1 else "SURVIVED"
        sg_los = sg.get('los_hours')
        sg_los_str = f"{sg_los:.1f}" if isinstance(sg_los, (int, float)) else "N/A"
        lines.append(f"\n[Similar Patient {i}] — Outcome: {outcome}")
        lines.append(f"  Diagnoses  : {', '.join(sg.get('diagnoses', [])) or 'None'}")
        lines.append(f"  Medications: {', '.join(sg.get('medications', [])) or 'None'}")
        lines.append(f"  Procedures : {', '.join(sg.get('procedures', [])) or 'None'}")
        lines.append(f"  LOS (hrs)  : {sg_los_str}")

    return "\n".join(lines)


# ---------------------------------------------------------
# 4. Llama-3 via Groq API (with retry on rate limit)
# ---------------------------------------------------------
def call_llama3(groq_client: Groq, context: str, gnn_prediction: str, max_retries: int = 3) -> str:
    system_prompt = (
        "You are a clinical decision support assistant. "
        "You are given structured information about a hospitalized patient "
        "and a list of clinically similar past patients with known outcomes. "
        "Your task is to explain, in clear medical language, why the predictive model "
        "predicted the given outcome for this patient, referencing the clinical evidence provided. "
        "Be concise, factual, and do not invent information not present in the context."
    )

    user_prompt = f"""
The GNN model predicted: **{gnn_prediction}**

Here is the clinical context retrieved from the Medical Knowledge Graph:

{context}

Based on this evidence, explain in 3-5 sentences why this prediction is clinically plausible,
referencing specific diagnoses, medications, procedures, or patterns from similar patients.
"""

    for attempt in range(1, max_retries + 1):
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"   Groq API attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(5 * attempt)  # exponential backoff
            else:
                return f"[Justification unavailable — Groq API error after {max_retries} attempts: {e}]"


# ---------------------------------------------------------
# 5. Load pre-computed GNN predictions (fast path)
# ---------------------------------------------------------
def load_gnn_predictions(proc_dir: str, data, device) -> torch.Tensor:
    """
    FIX: Load pre-computed predictions from CSV if available,
    instead of reloading the full graph + model on every RAG call.
    Falls back to recomputing if the CSV doesn't exist yet.
    """
    pred_cache_path = os.path.join(proc_dir, 'admission_predictions.csv')

    if os.path.exists(pred_cache_path):
        print("   Loading pre-computed GNN predictions from cache...")
        df_pred = pd.read_csv(pred_cache_path, index_col='admission_idx')
        probs = torch.tensor(df_pred['prob_death'].values, dtype=torch.float)
        return probs

    print("   No prediction cache found — running inference (this may take a moment)...")
    import torch_geometric.transforms

    data_cpu = torch_geometric.transforms.ToUndirected()(data)

    # Rebuild model
    model = MedicalKnowledgeGraphModel_Inference(data_cpu).to(device)
    model.load_state_dict(
        torch.load(os.path.join(proc_dir, 'heterosage_mortality.pth'),
                   map_location=device, weights_only=True)
    )
    model.eval()
    data_cpu = data_cpu.to(device)
    with torch.no_grad():
        out   = model(data_cpu.x_dict, data_cpu.edge_index_dict)['Admission']
        probs = torch.softmax(out, dim=1)[:, 1].cpu()

    # Cache to CSV so future calls skip the model reload
    df_cache = pd.DataFrame({'admission_idx': range(len(probs)), 'prob_death': probs.numpy()})
    df_cache.to_csv(pred_cache_path, index=False)
    print(f"   Predictions cached to {pred_cache_path}")

    return probs


# ---------------------------------------------------------
# 6. Main Pipeline
# ---------------------------------------------------------
def run_graph_rag(query_hadm_id: int = None, query_admission_idx: int = None, k: int = 5):
    """
    Full RAG pipeline for one patient admission.

    Args:
        query_hadm_id        : the real MIMIC hadm_id of the patient to explain
        query_admission_idx  : the row index in the embedding matrix (from script 06)
        k                    : number of similar patients to retrieve
    """
    load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
    config = load_config()
    proc_dir = config['paths']['processed_data']

    print("1. Loading admission embeddings...")
    embeddings = torch.load(os.path.join(proc_dir, 'admission_embeddings.pt'), weights_only=False)
    print(f"   Shape: {embeddings.shape}")

    df_adm = pd.read_csv(os.path.join(proc_dir, 'cleaned_admissions.csv'))
    hadm_to_idx = {hadm: idx for idx, hadm in enumerate(df_adm['hadm_id'].values)}
    idx_to_hadm = {idx: hadm for hadm, idx in hadm_to_idx.items()}

    if query_hadm_id is not None:
        query_admission_idx = hadm_to_idx.get(query_hadm_id)
        if query_admission_idx is None:
            raise ValueError(f"hadm_id {query_hadm_id} not found in cleaned_admissions.csv")
    elif query_admission_idx is None:
        data_tmp = torch.load(os.path.join(proc_dir, 'mimic_graph.pt'), weights_only=False)
        test_indices = data_tmp['Admission'].test_mask.nonzero(as_tuple=False).squeeze().tolist()
        query_admission_idx = test_indices[0]
        query_hadm_id = idx_to_hadm[query_admission_idx]
        print(f"   No query specified — using first test admission: hadm_id={query_hadm_id}")

    print(f"\n2. Finding {k} most similar admissions (FAISS)...")
    retriever = FAISSRetriever(embeddings)
    distances, similar_indices = retriever.find_similar(query_admission_idx, k=k)
    similar_hadm_ids = [int(idx_to_hadm[i]) for i in similar_indices]
    print(f"   Similar hadm_ids: {similar_hadm_ids}")

    print("\n3. Retrieving subgraphs from Neo4j...")
    neo4j_creds  = config['neo4j']
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo_retriever = Neo4jSubgraphRetriever(neo4j_creds['uri'], neo4j_creds['user'], neo4j_password)

    try:
        query_subgraph    = neo_retriever.get_admission_subgraph(int(query_hadm_id))
        similar_subgraphs = [neo_retriever.get_admission_subgraph(h) for h in similar_hadm_ids]
    finally:
        neo_retriever.close()

    print("\n4. Loading GNN prediction (cached fast path)...")
    data    = torch.load(os.path.join(proc_dir, 'mimic_graph.pt'), weights_only=False)
    device  = torch.device('cpu')
    probs   = load_gnn_predictions(proc_dir, data, device)

    pred_label = int(probs[query_admission_idx] >= 0.5)
    pred_prob  = probs[query_admission_idx].item()

    gnn_prediction = (
        f"{'HIGH MORTALITY RISK' if pred_label == 1 else 'LOW MORTALITY RISK'} "
        f"(confidence: {pred_prob:.1%})"
    )
    print(f"   GNN Prediction: {gnn_prediction}")

    print("\n5. Building RAG context...")
    context = build_rag_context(similar_subgraphs, query_subgraph)

    print("\n6. Calling Llama-3 via Groq for clinical justification...")
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in .env file!")
    groq_client   = Groq(api_key=groq_api_key)
    justification = call_llama3(groq_client, context, gnn_prediction)

    result = {
        "hadm_id":          int(query_hadm_id),
        "gnn_prediction":   gnn_prediction,
        "similar_patients": similar_hadm_ids,
        "justification":    justification,
    }

    print("\n" + "=" * 60)
    print(f"PATIENT     : hadm_id={query_hadm_id}")
    print(f"PREDICTION  : {gnn_prediction}")
    print(f"\nJUSTIFICATION:\n{justification}")
    print("=" * 60)

    output_path = os.path.join(proc_dir, f'rag_result_{query_hadm_id}.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"\nResult saved to: {output_path}")

    return result


# ---------------------------------------------------------
# Model wrapper (mirrors architecture from 06 exactly)
# ---------------------------------------------------------
class MedicalKnowledgeGraphModel_Inference(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        from torch_geometric.nn import SAGEConv, to_hetero

        HIDDEN = 64

        class FeatureEncoderLocal(torch.nn.Module):
            def __init__(self, hidden, data):
                super().__init__()
                self.pat_lin  = torch.nn.Linear(data['Patient'].x.shape[1],   hidden)
                self.adm_lin  = torch.nn.Linear(data['Admission'].x.shape[1], hidden)
                self.diag_emb = torch.nn.Embedding(data['Diagnosis'].num_nodes,  hidden)
                self.med_emb  = torch.nn.Embedding(data['Medication'].num_nodes, hidden)
                self.proc_emb = torch.nn.Embedding(data['Procedure'].num_nodes,  hidden)
                self.lab_emb  = torch.nn.Embedding(data['LabTest'].num_nodes,    hidden)
                self.pat_bn   = torch.nn.BatchNorm1d(hidden)
                self.adm_bn   = torch.nn.BatchNorm1d(hidden)

            def forward(self, x_dict):
                return {
                    'Patient':    self.pat_bn(self.pat_lin(x_dict['Patient'])),
                    'Admission':  self.adm_bn(self.adm_lin(x_dict['Admission'])),
                    'Diagnosis':  self.diag_emb.weight,
                    'Medication': self.med_emb.weight,
                    'Procedure':  self.proc_emb.weight,
                    'LabTest':    self.lab_emb.weight,
                }

        class BaseGNNLocal(torch.nn.Module):
            def __init__(self, h, out):
                super().__init__()
                self.conv1   = SAGEConv(h, h)
                self.conv2   = SAGEConv(h, h)
                self.conv3   = SAGEConv(h, h)
                self.dropout = torch.nn.Dropout(0.3)
                self.bn1     = torch.nn.BatchNorm1d(h)
                self.bn2     = torch.nn.BatchNorm1d(h)
                self.lin     = torch.nn.Linear(h, out)

            def forward(self, x, edge_index):
                x = self.bn1(self.conv1(x, edge_index)).relu()
                x = self.dropout(x)
                x = self.bn2(self.conv2(x, edge_index)).relu()
                x = self.dropout(x)
                return self.lin(self.conv3(x, edge_index).relu())

        self.encoder = FeatureEncoderLocal(HIDDEN, data)
        self.gnn     = to_hetero(BaseGNNLocal(HIDDEN, 2), data.metadata(), aggr='max')

    def forward(self, x_dict, edge_index_dict):
        return self.gnn(self.encoder(x_dict), edge_index_dict)


if __name__ == "__main__":
    run_graph_rag()