import pandas as pd
import yaml
import os
import json
from neo4j import GraphDatabase
from tqdm import tqdm
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def load_config():
    config_path = os.path.join(PROJECT_ROOT, 'config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


class MIMICGraphBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_database(self):
        print(" -> Clearing existing database (batch delete to avoid OOM)...")
        query = """
        MATCH (n)
        WITH n LIMIT 10000
        DETACH DELETE n
        RETURN count(n) as deleted_count
        """
        with self.driver.session() as session:
            while True:
                result = session.run(query)
                deleted = result.single()["deleted_count"]
                if deleted == 0:
                    break

    def create_constraints(self):
        print(" -> Creating constraints and indexes...")
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Patient)   REQUIRE p.id       IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Admission)  REQUIRE a.id       IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Diagnosis)  REQUIRE d.code     IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Medication) REQUIRE m.name     IS UNIQUE",
            # NEW
            "CREATE CONSTRAINT IF NOT EXISTS FOR (pr:Procedure) REQUIRE pr.code    IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (l:LabTest)    REQUIRE l.item_id  IS UNIQUE",
        ]
        with self.driver.session() as session:
            for q in queries:
                session.run(q)

    def load_patients_and_admissions(self, df_pat, df_adm):
        print(" -> Injecting Patients and Admissions...")
        df_merged = df_adm.merge(df_pat, on='subject_id', how='left')
        df_merged = df_merged.where(pd.notnull(df_merged), None)
        records = df_merged.to_dict('records')

        query = """
        UNWIND $records AS row
        MERGE (p:Patient {id: row.subject_id})
        ON CREATE SET
            p.gender     = row.gender,
            p.anchor_age = row.anchor_age

        MERGE (a:Admission {id: row.hadm_id})
        ON CREATE SET
            a.admittime       = row.admittime,
            a.dischtime       = row.dischtime,
            a.admission_type  = row.admission_type,
            a.insurance       = row.insurance,
            a.marital_status  = row.marital_status,
            a.race            = row.race,
            a.los_hours       = row.los_hours,
            a.hospital_expire_flag = row.hospital_expire_flag

        MERGE (p)-[:HAS_ADMISSION]->(a)
        """
        batch_size = 10000
        with self.driver.session() as session:
            for i in tqdm(range(0, len(records), batch_size), desc="Patients & Admissions"):
                session.run(query, records=records[i:i + batch_size])

    def load_diagnoses(self, df_diag):
        print(" -> Injecting Diagnoses...")
        records = df_diag.where(pd.notnull(df_diag), None).to_dict('records')
        query = """
        UNWIND $records AS row
        MATCH (a:Admission {id: row.hadm_id})
        MERGE (d:Diagnosis {code: row.icd_code})
        ON CREATE SET d.icd_version = row.icd_version
        MERGE (a)-[:DIAGNOSED_WITH]->(d)
        """
        batch_size = 10000
        with self.driver.session() as session:
            for i in tqdm(range(0, len(records), batch_size), desc="Diagnoses"):
                session.run(query, records=records[i:i + batch_size])

    def load_prescriptions(self, df_presc):
        print(" -> Injecting Medications...")
        df_presc = df_presc.dropna(subset=['starttime', 'drug'])
        df_presc = df_presc.where(pd.notnull(df_presc), None)
        records = df_presc.to_dict('records')
        query = """
        UNWIND $records AS row
        MATCH (a:Admission {id: row.hadm_id})
        MERGE (m:Medication {name: row.drug})
        MERGE (a)-[:PRESCRIBED {start_time: row.starttime}]->(m)
        """
        batch_size = 10000
        with self.driver.session() as session:
            for i in tqdm(range(0, len(records), batch_size), desc="Medications"):
                session.run(query, records=records[i:i + batch_size])

    # ---------------------------------------------------------
    # NEW: Procedures
    # ---------------------------------------------------------
    def load_procedures(self, df_proc):
        print(" -> Injecting Procedures...")
        df_proc = df_proc.where(pd.notnull(df_proc), None)
        records = df_proc.to_dict('records')
        query = """
        UNWIND $records AS row
        MATCH (a:Admission {id: row.hadm_id})
        MERGE (pr:Procedure {code: row.icd_code})
        ON CREATE SET pr.icd_version = row.icd_version
        MERGE (a)-[:HAS_PROCEDURE]->(pr)
        """
        batch_size = 10000
        with self.driver.session() as session:
            for i in tqdm(range(0, len(records), batch_size), desc="Procedures"):
                session.run(query, records=records[i:i + batch_size])

    # ---------------------------------------------------------
    # NEW: Lab Events (aggregated per admission per test)
    # ---------------------------------------------------------
    def load_labevents(self, df_lab):
        print(" -> Injecting Lab Results (aggregated per admission)...")
        # Aggregate per (hadm_id, itemid): mean + std of values
        agg = df_lab.groupby(['hadm_id', 'itemid'])['valuenum'].agg(['mean', 'std']).reset_index()
        agg.columns = ['hadm_id', 'item_id', 'mean_value', 'std_value']
        agg = agg.where(pd.notnull(agg), None)
        records = agg.to_dict('records')

        query = """
        UNWIND $records AS row
        MATCH (a:Admission {id: row.hadm_id})
        MERGE (l:LabTest {item_id: row.item_id})
        MERGE (a)-[r:HAS_LAB]->(l)
        SET r.mean_value = row.mean_value,
            r.std_value  = row.std_value
        """
        batch_size = 10000
        with self.driver.session() as session:
            for i in tqdm(range(0, len(records), batch_size), desc="Lab Events"):
                session.run(query, records=records[i:i + batch_size])

    # ---------------------------------------------------------
    # NLP Metadata (now includes medications from NER)
    # ---------------------------------------------------------
    def inject_nlp_metadata(self, json_data):
        print(" -> Injecting NLP metadata (symptoms, diseases, medications)...")
        records = []
        for hadm_id, data in json_data.items():
            records.append({
                "hadm_id":      int(hadm_id),
                "nlp_symptoms":    data.get("symptoms",    []),
                "nlp_diseases":    data.get("diseases",    []),
                "nlp_medications": data.get("medications", []),  # NEW
            })
        query = """
        UNWIND $records AS row
        MATCH (a:Admission {id: row.hadm_id})
        SET a.extracted_symptoms    = row.nlp_symptoms,
            a.extracted_diseases    = row.nlp_diseases,
            a.extracted_medications = row.nlp_medications
        """
        batch_size = 10000
        with self.driver.session() as session:
            for i in tqdm(range(0, len(records), batch_size), desc="NLP Metadata"):
                session.run(query, records=records[i:i + batch_size])

    # ---------------------------------------------------------
    # NEW: Graph Validation (Phase 6)
    # ---------------------------------------------------------
    def validate_graph(self):
        print("\n -> Running graph validation (Phase 6)...")
        checks = {
            "Orphan Admissions (no Patient)": """
                MATCH (a:Admission)
                WHERE NOT (:Patient)-[:HAS_ADMISSION]->(a)
                RETURN count(a) AS count
            """,
            "Orphan Diagnoses (no Admission)": """
                MATCH (d:Diagnosis)
                WHERE NOT (:Admission)-[:DIAGNOSED_WITH]->(d)
                RETURN count(d) AS count
            """,
            "Orphan Procedures (no Admission)": """
                MATCH (pr:Procedure)
                WHERE NOT (:Admission)-[:HAS_PROCEDURE]->(pr)
                RETURN count(pr) AS count
            """,
            "Admissions with negative LOS": """
                MATCH (a:Admission)
                WHERE a.los_hours < 0
                RETURN count(a) AS count
            """,
        }
        validation_report = {}
        with self.driver.session() as session:
            for check_name, query in checks.items():
                result = session.run(query).single()
                count = result["count"]
                status = "OK" if count == 0 else f"WARNING: {count} found"
                print(f"    [{status}] {check_name}")
                validation_report[check_name] = count

        return validation_report


def build_knowledge_graph():
    load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    if not neo4j_password:
        raise ValueError("Neo4j password not found in .env file!")

    config = load_config()
    proc_dir = os.path.join(PROJECT_ROOT, config['paths']['processed_data'])

    print("1. Loading CSV data into Pandas...")
    df_pat   = pd.read_csv(os.path.join(proc_dir, 'cleaned_patients.csv'))
    df_adm   = pd.read_csv(os.path.join(proc_dir, 'cleaned_admissions.csv'))
    df_diag  = pd.read_csv(os.path.join(proc_dir, 'cleaned_diagnoses.csv'))
    df_presc = pd.read_csv(os.path.join(proc_dir, 'cleaned_prescriptions.csv'))
    df_proc  = pd.read_csv(os.path.join(proc_dir, 'cleaned_procedures.csv'))   # NEW
    df_lab   = pd.read_csv(os.path.join(proc_dir, 'cleaned_labevents.csv'))    # NEW

    with open(os.path.join(proc_dir, 'nlp_enriched_properties.json'), 'r', encoding='utf-8') as f:
        nlp_data = json.load(f)

    print("\n2. Connecting to Neo4j...")
    neo4j_creds = config['neo4j']
    builder = MIMICGraphBuilder(neo4j_creds['uri'], neo4j_creds['user'], neo4j_password)

    try:
        builder.clear_database()
        builder.create_constraints()

        print("\n3. Building graph topology...")
        builder.load_patients_and_admissions(df_pat, df_adm)
        builder.load_diagnoses(df_diag)
        builder.load_prescriptions(df_presc)
        builder.load_procedures(df_proc)   # NEW
        builder.load_labevents(df_lab)     # NEW

        print("\n4. Semantic enrichment...")
        builder.inject_nlp_metadata(nlp_data)

        print("\n5. Graph Validation (Phase 6)...")
        report = builder.validate_graph()

        import json as _json
        with open(os.path.join(proc_dir, 'stats_phase6_validation.json'), 'w') as f:
            _json.dump(report, f, indent=4)

        print("\nPhases 5 & 6 Complete! Open Neo4j Browser to explore your Knowledge Graph.")

    finally:
        builder.close()


if __name__ == "__main__":
    build_knowledge_graph()