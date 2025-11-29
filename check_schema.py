import os
import pymysql
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("check_schema")

# Config from main.py
class Config:
    DB_HOST = os.getenv("DB_HOST", "smartvoting-db.cpkg8mew2xde.ap-south-2.rds.amazonaws.com")
    DB_PORT = int(os.getenv("DB_PORT", "3306"))
    DB_USER = os.getenv("DB_USER", "admin")
    DB_PASS = os.getenv("DB_PASS", "Likhith2411")
    DB_NAME = os.getenv("DB_NAME", "smartvoting")

def check_schema():
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
        with conn.cursor() as cur:
            logger.info("DESCRIBE nominee:")
            cur.execute("DESCRIBE nominee")
            for row in cur.fetchall():
                logger.info(row)

            logger.info("DESCRIBE vote:")
            cur.execute("DESCRIBE vote")
            for row in cur.fetchall():
                logger.info(row)
                
            logger.info("SHOW CREATE TABLE vote:")
            cur.execute("SHOW CREATE TABLE vote")
            row = cur.fetchone()
            logger.info(row['Create Table'])

    except Exception as e:
        logger.error(f"Schema check failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_schema()
