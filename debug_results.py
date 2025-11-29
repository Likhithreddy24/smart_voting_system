import os
import pymysql
import logging
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_results")

# Config from main.py
class Config:
    DB_HOST = os.getenv("DB_HOST", "smartvoting-db.cpkg8mew2xde.ap-south-2.rds.amazonaws.com")
    DB_PORT = int(os.getenv("DB_PORT", "3306"))
    DB_USER = os.getenv("DB_USER", "admin")
    DB_PASS = os.getenv("DB_PASS", "Likhith2411")
    DB_NAME = os.getenv("DB_NAME", "smartvoting")

def read_sql(query, conn, params=None):
    with conn.cursor() as cur:
        cur.execute(query, params)
        result = cur.fetchall()
    return pd.DataFrame(result)

def test_results():
    logger.info("Connecting to DB...")
    try:
        conn = pymysql.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            user=Config.DB_USER,
            password=Config.DB_PASS,
            database=Config.DB_NAME,
            cursorclass=pymysql.cursors.DictCursor
        )
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return

    try:
        # 1. Fetch Data
        logger.info("Fetching nominees...")
        df_nom = read_sql('SELECT * FROM nominee', conn)
        logger.info(f"Nominees: {len(df_nom)}")
        
        logger.info("Fetching votes...")
        votes_df = read_sql('SELECT * FROM vote', conn)
        logger.info(f"Votes: {len(votes_df)}")
        
        if not votes_df.empty:
            logger.info(f"Sample Vote: {votes_df.iloc[0].to_dict()}")

        # 2. Simulate Calculation (Current Buggy Logic)
        logger.info("--- Simulating Current Logic ---")
        vote_counts = {}
        if not votes_df.empty:
            vote_counts = votes_df['vote'].value_counts().to_dict()
        
        logger.info(f"Vote Counts (Keys are IDs): {vote_counts}")
        
        for index, row in df_nom.iterrows():
            symbol = str(row['symbol_name'])
            # FIXED: Looking up by ID
            count = vote_counts.get(row['id'], 0)
            logger.info(f"Nominee: {row['member_name']} (ID={row['id']}, Symbol={symbol}) -> Count: {count}")
            
            if count == 0 and row['id'] in vote_counts:
                logger.error(f"MISMATCH! ID {row['id']} has {vote_counts[row['id']]} votes, but lookup got 0.")
            elif count > 0:
                logger.info("MATCH! Count retrieved successfully.")

        # 3. Simulate Fix
        logger.info("--- Simulating Fix ---")
        for index, row in df_nom.iterrows():
            nom_id = row['id']
            # FIX: Look up by ID
            count = vote_counts.get(nom_id, 0)
            logger.info(f"Nominee: {row['member_name']} (ID={row['id']}) -> Count: {count}")

    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    test_results()
