import sqlite3
import pandas as pd

DB_PATH = "src/Data/database.db"
conn = sqlite3.connect(DB_PATH)

tables = ['cameras', 'weapon_detections', 'violence_detections', 'criminal_recognitions', 'criminals']

for t in tables:
    df = pd.read_sql_query(f"SELECT * FROM {t}", conn)
    print(f"\n=== Table: {t} ===")
    # Print entire DataFrame without truncation
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

conn.close()
