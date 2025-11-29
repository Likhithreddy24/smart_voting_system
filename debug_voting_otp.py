import os
import pymysql
import datetime
import logging
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_voting_otp")

# Config from main.py
class Config:
    DB_HOST = os.getenv("DB_HOST", "smartvoting-db.cpkg8mew2xde.ap-south-2.rds.amazonaws.com")
    DB_PORT = int(os.getenv("DB_PORT", "3306"))
    DB_USER = os.getenv("DB_USER", "admin")
    DB_PASS = os.getenv("DB_PASS", "Likhith2411")
    DB_NAME = os.getenv("DB_NAME", "smartvoting")

def test_voting_otp():
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

    voter_id = f"TEST_VOTE_{random.randint(1000,9999)}"
    otp_code = "654321"
    email = "test_vote@example.com"
    
    try:
        with conn.cursor() as cur:
            # 1. Setup: Create an existing voter (simulating registration completed)
            logger.info(f"Creating existing voter {voter_id}...")
            cur.execute("DELETE FROM voters WHERE voter_id=%s", (voter_id,))
            cur.execute("""
                INSERT INTO voters (first_name, voter_id, email, verified, aadhar_id)
                VALUES ('TestVoter', %s, %s, 'yes', %s)
            """, (voter_id, email, f"999{random.randint(100000000,999999999)}"))
            
            # 2. Simulate Login (Generate OTP)
            logger.info("Simulating Login (Generating OTP)...")
            expiry = datetime.datetime.now() + datetime.timedelta(minutes=5)
            
            # Check if columns exist (Implicitly testing schema update)
            try:
                cur.execute("""
                    UPDATE voters 
                    SET otp_code=%s, otp_expires_at=%s 
                    WHERE voter_id=%s
                """, (otp_code, expiry, voter_id))
                logger.info("OTP stored in voters table.")
            except Exception as e:
                logger.error(f"Failed to update voters table. Columns missing? {e}")
                return

            # 3. Simulate Verify (Check OTP)
            logger.info("Simulating Verification...")
            now_time = datetime.datetime.now()
            
            # Logic from main.py (Active Voter Check)
            cur.execute("""
                SELECT * FROM voters
                WHERE voter_id=%s AND otp_code=%s AND otp_expires_at > %s
            """, (voter_id, otp_code, now_time))
            voter = cur.fetchone()
            
            if voter:
                logger.info("Verification SUCCESS (Active Voter).")
                # Cleanup
                cur.execute("UPDATE voters SET otp_code=NULL, otp_expires_at=NULL WHERE voter_id=%s", (voter_id,))
            else:
                logger.error("Verification FAILED.")
                
                # Debug
                cur.execute("SELECT * FROM voters WHERE voter_id=%s", (voter_id,))
                rec = cur.fetchone()
                logger.info(f"Record: OTP={rec.get('otp_code')}, Expires={rec.get('otp_expires_at')}")

    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    test_voting_otp()
