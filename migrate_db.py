import os
import pymysql
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("migrate_db")

# Config from main.py
class Config:
    DB_HOST = os.getenv("DB_HOST", "smartvoting-db.cpkg8mew2xde.ap-south-2.rds.amazonaws.com")
    DB_PORT = int(os.getenv("DB_PORT", "3306"))
    DB_USER = os.getenv("DB_USER", "admin")
    DB_PASS = os.getenv("DB_PASS", "Likhith2411")
    DB_NAME = os.getenv("DB_NAME", "smartvoting")

def migrate():
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

    try:
        with conn.cursor() as cur:
            logger.info("Adding otp_code column...")
            try:
                cur.execute("ALTER TABLE voters ADD COLUMN otp_code VARCHAR(10)")
                logger.info("Added otp_code.")
            except Exception as e:
                logger.warning(f"otp_code might exist: {e}")

            logger.info("Adding otp_expires_at column...")
            try:
                cur.execute("ALTER TABLE voters ADD COLUMN otp_expires_at DATETIME")
                logger.info("Added otp_expires_at.")
            except Exception as e:
                logger.warning(f"otp_expires_at might exist: {e}")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
