import pandas as pd
from main import get_db_connection

try:
    mydb = get_db_connection()
    df = pd.read_sql_query('SELECT * FROM vote', mydb)
    print("Total votes:", len(df))
    print(df)
except Exception as e:
    print("Error:", e)
