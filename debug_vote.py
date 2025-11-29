import os
import pymysql
import logging
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_vote")

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

def test_vote():
    logger.info("Connecting to DB...")
    try:
        conn = pymysql.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            user=Config.DB_USER,
            password=Config.DB_PASS,
            database=Config.DB_NAME,
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True
        )
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return

    aadhar = "TEST_A_99999"
    
    try:
        # 1. Check Nominees
        logger.info("Fetching nominees...")
        df_nom = read_sql('SELECT * FROM nominee', conn)
        if df_nom.empty:
            logger.error("No nominees found! Voting will fail.")
            return
        
        all_nom = df_nom['symbol_name'].values
        logger.info(f"Nominees: {all_nom}")
        
        vote_choice = all_nom[0]
        logger.info(f"Selected choice: {vote_choice}")

        # 2. Setup: Create Voter to satisfy FK
        logger.info(f"Creating voter {aadhar}...")
        with conn.cursor() as cur:
            cur.execute("DELETE FROM vote WHERE aadhar=%s", (aadhar,))
            cur.execute("DELETE FROM voters WHERE aadhar_id=%s", (aadhar,))
            cur.execute("""
                INSERT INTO voters (first_name, voter_id, email, verified, aadhar_id)
                VALUES ('TestVote', 'V_TEST_999', 'test_vote@example.com', 'yes', %s)
            """, (aadhar,))

        # 3. Check if already voted (Pre-check)
        logger.info(f"Checking if {aadhar} voted...")
        df_exists = read_sql('SELECT 1 FROM vote WHERE aadhar=%s LIMIT 1', conn, params=[aadhar])
        if not df_exists.empty:
            logger.info("User already voted (Pre-check). Clearing for test...")
            with conn.cursor() as cur:
                cur.execute("DELETE FROM vote WHERE aadhar=%s", (aadhar,))
        else:
            logger.info("User has not voted.")

        # 4. Simulate Vote Insertion (FIXED LOGIC)
        logger.info("Inserting vote...")
        
        # Look up ID
        df_selected = read_sql('SELECT id FROM nominee WHERE symbol_name=%s', conn, params=[vote_choice])
        if df_selected.empty:
            logger.error("Nominee ID lookup failed!")
            return
        vote_id = int(df_selected.iloc[0]['id'])
        logger.info(f"Resolved {vote_choice} to ID {vote_id}")

        sql = "INSERT INTO vote (vote, aadhar) VALUES (%s, %s)"
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (vote_id, aadhar))
            # conn.commit() # Autocommit is True
            logger.info("Vote inserted successfully.")
        except Exception as e:
            logger.error(f"Insert failed: {e}")

        # 4. Verify Insertion
        df_check = read_sql('SELECT * FROM vote WHERE aadhar=%s', conn, params=[aadhar])
        if not df_check.empty:
            logger.info(f"Verification: Vote found for {aadhar}: {df_check.iloc[0]['vote']}")
        else:
            logger.error("Verification: Vote NOT found!")

    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    test_vote()
