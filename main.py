import os
import re
import io
import time
import datetime
import shutil
import logging
import pickle
import zipfile
import tempfile
from typing import Optional, Tuple, Dict, Any, List, Deque
from collections import deque, defaultdict
import requests
import random
import datetime
import time
import logging
from flask import (
    Flask, render_template, url_for, request, session, flash, redirect,
    send_file, abort
)
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

import pymysql
import pandas as pd
import numpy as np
import cv2
from PIL import Image

# -------------------------
# Configuration
# -------------------------

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev")
    DB_HOST = os.getenv("DB_HOST", "smartvoting-db.cpkg8mew2xde.ap-south-2.rds.amazonaws.com")
    DB_PORT = int(os.getenv("DB_PORT", "3306"))
    DB_USER = os.getenv("DB_USER", "admin")
    DB_PASS = os.getenv("DB_PASS", "Likhith2411")
    DB_NAME = os.getenv("DB_NAME", "smartvoting")        # 👈 correct database name

    # Mail (for OTP)
    MAIL_USER = os.environ.get("MAIL_USER", "likhithreddygg@gmail.com")
    MAIL_PASS = os.environ.get("MAIL_PASS", "ltindlsamolfsjoc")

    # SMS (Fast2SMS)
    FAST2SMS_API_KEY = os.environ.get("FAST2SMS_API_KEY", "")

    # Face recognition
    MIN_IMAGES_PER_LABEL = int(os.environ.get("MIN_IMAGES_PER_LABEL", "15"))
    BLUR_THRESHOLD = float(os.environ.get("BLUR_THRESHOLD", "60.0"))
    MIN_FACE_SIZE = int(os.environ.get("MIN_FACE_SIZE", "100"))

    # Voting capture limits
    VOTING_MAX_SECONDS = int(os.environ.get("VOTING_MAX_SECONDS", "15"))
    VOTING_MAX_FRAMES = int(os.environ.get("VOTING_MAX_FRAMES", "500"))

    # Voting recognition tuning
    LBPH_CONF_THRESHOLD = float(os.environ.get("LBPH_CONF_THRESHOLD", "80"))  # not used directly now
    VOTING_MIN_FACE_SIZE = int(os.environ.get("VOTING_MIN_FACE_SIZE", os.environ.get("MIN_FACE_SIZE", "70")))
    VOTING_SCALE_FACTOR = float(os.environ.get("VOTING_SCALE_FACTOR", "1.1"))
    VOTING_MIN_NEIGHBORS = int(os.environ.get("VOTING_MIN_NEIGHBORS", "4"))
    ROI_SIZE = int(os.environ.get("ROI_SIZE", "200"))  # normalize face ROI to this square size

    # New strict voting parameters
    VOTING_STRICT_CONF_THRESHOLD = float(os.environ.get("VOTING_STRICT_CONF_THRESHOLD", "60.0"))
    VOTING_REQUIRED_CONSEC_MATCHES = int(os.environ.get("VOTING_REQUIRED_CONSEC_MATCHES", "7"))

    # OTP
    OTP_EXPIRY_SECONDS = int(os.environ.get("OTP_EXPIRY_SECONDS", "300"))  # 5 minutes
    OTP_RESEND_MIN_INTERVAL = int(os.environ.get("OTP_RESEND_MIN_INTERVAL", "60"))  # 60s

    # Admin IP allowlist (comma-separated), optional
    ADMIN_IP_ALLOWLIST = os.environ.get("ADMIN_IP_ALLOWLIST", "")

    # Capture strictness (relaxed)
    CAPTURE_STABLE_CONSEC_FRAMES = int(os.environ.get("CAPTURE_STABLE_CONSEC_FRAMES", "8"))   # was 15
    CAPTURE_MIN_FACE_RATIO = float(os.environ.get("CAPTURE_MIN_FACE_RATIO", "0.15"))          # Reduced from 0.30 to 0.15 for easier capture
    CAPTURE_CENTER_TOLERANCE = float(os.environ.get("CAPTURE_CENTER_TOLERANCE", "0.35"))      # Increased from 0.25 to 0.35
    CAPTURE_LBPH_CONF_THRESHOLD = float(os.environ.get("CAPTURE_LBPH_CONF_THRESHOLD", "30.0"))  # lowered to reduce false positives

config = Config()

# -------------------------
# App & Logging
# -------------------------

app = Flask(__name__)
app.debug = True
app.config['SECRET_KEY'] = config.SECRET_KEY

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("smart_voting")

# -------------------------
# Security: secret checks (only enforce in non-debug)
# -------------------------

def _require_secrets_in_production():
    if app.debug:
        return
    missing = []
    if not os.environ.get("SECRET_KEY") or os.environ.get("SECRET_KEY") == "dev-secret-key-change-me":
        missing.append("SECRET_KEY")
    if not os.environ.get("DB_PASS") or os.environ.get("DB_PASS") == "Likhith@24":
        missing.append("DB_PASS")
    if not os.environ.get("MAIL_USER") or not os.environ.get("MAIL_PASS"):
        logger.warning("MAIL_USER/MAIL_PASS not set. OTP emails will fail in production.")
    if missing:
        raise RuntimeError(f"Missing/unsafe secrets in production: {', '.join(missing)}")

# -------------------------
# Database
# -------------------------

def get_db_connection():
    conn = pymysql.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        user=Config.DB_USER,
        password=Config.DB_PASS,
        database=Config.DB_NAME,  # smartvoting
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
        # ssl={"ssl": ssl.create_default_context()}  # if you require SSL
    )
    with conn.cursor() as cur:
        cur.execute("USE smartvoting")  # ✅ force the schema
    return conn

def ensure_db_connection():
    global mydb
    try:
        mydb.ping(reconnect=True)
    except Exception:
        try:
            mydb.close()
        except Exception:
            pass
        mydb = get_db_connection()

    # Ensure voters table has OTP columns (Schema Migration)
    try:
        with mydb.cursor() as cur:
            cur.execute("ALTER TABLE voters ADD COLUMN otp_code VARCHAR(10)")
            cur.execute("ALTER TABLE voters ADD COLUMN otp_expires_at DATETIME")
    except Exception:
        pass # Columns likely exist

# Keep a global connection for pandas read_sql_query compatibility
mydb = get_db_connection()

def read_sql(query, conn, params=None):
    """
    Replacement for pd.read_sql_query that works with DictCursor.
    """
    with conn.cursor() as cur:
        cur.execute(query, params)
        result = cur.fetchall()
    return pd.DataFrame(result)

# -------------------------
# OpenCV & Face models
# -------------------------

facedata = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(facedata)

# In-memory caches
_recognizer_cache: Optional['cv2.face_LBPHFaceRecognizer'] = None  # type: ignore
_encoder_cache: Optional[Any] = None
_model_mtime: Optional[float] = None
_encoder_mtime: Optional[float] = None

def has_cv2_face() -> bool:
    return hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create")

def _file_mtime(path: str) -> Optional[float]:
    try:
        return os.path.getmtime(path)
    except Exception:
        return None

def load_encoder(force: bool = False):
    global _encoder_cache, _encoder_mtime
    path = "encoder.pkl"
    mtime = _file_mtime(path)
    if _encoder_cache is None or force or (_encoder_mtime is None or (mtime and mtime != _encoder_mtime)):
        if not os.path.exists(path):
            _encoder_cache = None
            _encoder_mtime = None
            return None
        with open(path, "rb") as f:
            _encoder_cache = pickle.load(f)
        _encoder_mtime = mtime
        logger.info("Encoder loaded. Classes: %s", list(getattr(_encoder_cache, "classes_", [])))
    return _encoder_cache

def load_recognizer(force: bool = False):
    global _recognizer_cache, _model_mtime
    path = "Trained.yml"
    mtime = _file_mtime(path)
    if _recognizer_cache is None or force or (_model_mtime is None or (mtime and mtime != _model_mtime)):
        if not has_cv2_face():
            return None
        if not os.path.exists(path):
            _recognizer_cache = None
            _model_mtime = None
            return None
        rec = cv2.face.LBPHFaceRecognizer_create()
        rec.read(path)
        _recognizer_cache = rec
        _model_mtime = mtime
        logger.info("Recognizer loaded from %s", path)
    return _recognizer_cache

# -------------------------
# Recognition Observability
# -------------------------

RECENT_ATTEMPTS: Deque[Dict[str, Any]] = deque(maxlen=500)

def record_attempt(conf: Optional[float], accepted: bool, label: Optional[int]):
    RECENT_ATTEMPTS.append({
        "ts": time.time(),
        "conf": conf,
        "accepted": accepted,
        "label": label
    })

def summarize_attempts() -> Dict[str, Any]:
    total = len(RECENT_ATTEMPTS)
    accepted = sum(1 for a in RECENT_ATTEMPTS if a["accepted"])
    confs = [a["conf"] for a in RECENT_ATTEMPTS if a["conf"] is not None]
    per_label = defaultdict(int)
    for a in RECENT_ATTEMPTS:
        if a["label"] is not None:
            per_label[a["label"]] += 1
    return {
        "total": total,
        "accepted": accepted,
        "accept_rate": (accepted / total * 100.0) if total else 0.0,
        "conf_min": min(confs) if confs else None,
        "conf_max": max(confs) if confs else None,
        "conf_avg": (sum(confs) / len(confs)) if confs else None,
        "per_label_counts": dict(per_label)
    }

# -------------------------
# Helpers
# -------------------------

def try_open_camera() -> Optional[cv2.VideoCapture]:
    attempts = [
        (0, cv2.CAP_AVFOUNDATION),
        (0, 0),
        (1, 0),
    ]
    for idx, backend in attempts:
        try:
            cam = cv2.VideoCapture(idx, backend)
            if cam and cam.isOpened():
                return cam
            if cam:
                cam.release()
        except Exception:
            continue
    return None

def is_blurry(img_gray: np.ndarray) -> bool:
    fm = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    return fm < config.BLUR_THRESHOLD

def valid_email(email: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email.strip()))

def valid_phone(pno: str) -> bool:
    digits = re.sub(r"\D", "", pno)
    return 8 <= len(digits) <= 15

def send_otp_email(to_address: str, otp_code: str) -> bool:
    import os, threading, logging, time, requests, smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    logger = logging.getLogger(__name__)

    def _worker():
        try:
            # Try importing your config module if available
            try:
                import config
            except Exception:
                config = None

            # Determine Sender Address
            # 1. Prefer RESEND_FROM for Resend API (e.g., "onboarding@resend.dev" or "auth@yourdomain.com")
            resend_from = os.environ.get("RESEND_FROM")
            
            # 2. Fallback to MAIL_FROM or MAIL_USER for SMTP
            mail_from = os.environ.get("MAIL_FROM")
            if not mail_from:
                mail_from = (
                    (getattr(config, "MAIL_USER", None) if config else None)
                    or os.environ.get("MAIL_USER")
                    or os.environ.get("MAIL_USERNAME")
                    or "noreply@example.com"
                )

            # OTP expiry (from config or environment)
            if config and hasattr(config, "OTP_EXPIRY_SECONDS"):
                expiry_minutes = int(getattr(config, "OTP_EXPIRY_SECONDS")) // 60
            else:
                expiry_minutes = int(os.environ.get("OTP_EXPIRY_SECONDS", "300")) // 60

            # ---------------------------------------------------------
            # STRATEGY 1: Resend API (HTTP) - Works on Render Free Tier
            # ---------------------------------------------------------
            api_key = os.environ.get("RESEND_API_KEY")
            if api_key:
                # Use RESEND_FROM if set, otherwise fallback to mail_from
                sender = resend_from if resend_from else mail_from
                
                # Warning for generic emails with Resend
                if "@gmail.com" in sender or "@yahoo.com" in sender:
                    logger.warning(
                        "⚠️ Using a generic email (%s) with Resend may fail. "
                        "Use 'onboarding@resend.dev' (for testing) or a verified domain.", 
                        sender
                    )

                # Build email payload
                payload = {
                    "from": sender,
                    "to": [to_address],
                    "subject": "Your OTP Code for Smart Voting",
                    "html": (
                        f"<p>Your OTP is <b>{otp_code}</b>.</p>"
                        f"<p>This code will expire in {expiry_minutes} minutes.</p>"
                    ),
                }
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }

                # Try sending with up to 3 retries
                for attempt in range(3):
                    try:
                        resp = requests.post(
                            "https://api.resend.com/emails",
                            json=payload,
                            headers=headers,
                            timeout=15,
                        )
                        if 200 <= resp.status_code < 300:
                            logger.info("✅ OTP email dispatched to %s via Resend (Sender: %s)", to_address, sender)
                            return
                        else:
                            logger.error(
                                "Resend error (HTTP %s): %s", resp.status_code, resp.text
                            )
                    except Exception as e:
                        logger.error("Resend request failed (attempt %d): %r", attempt + 1, e)
                    time.sleep(1.5 * (attempt + 1))
                
                logger.warning("Resend failed after retries. Trying SMTP fallback...")

            # ---------------------------------------------------------
            # STRATEGY 2: SMTP (Standard) - May be blocked on Render Free
            # ---------------------------------------------------------
            mail_user = os.environ.get("MAIL_USER")
            mail_pass = os.environ.get("MAIL_PASS")
            mail_server = os.environ.get("MAIL_SERVER", "smtp.gmail.com")
            mail_port = int(os.environ.get("MAIL_PORT", "587"))

            if mail_user and mail_pass:
                msg = MIMEMultipart()
                msg['From'] = mail_from
                msg['To'] = to_address
                msg['Subject'] = "Your OTP Code for Smart Voting"

                body = (
                    f"Your OTP is {otp_code}.\n"
                    f"This code will expire in {expiry_minutes} minutes."
                )
                msg.attach(MIMEText(body, 'plain'))

                try:
                    server = smtplib.SMTP(mail_server, mail_port)
                    server.starttls()
                    server.login(mail_user, mail_pass)
                    text = msg.as_string()
                    server.sendmail(mail_from, to_address, text)
                    server.quit()
                    logger.info("✅ OTP email dispatched to %s via SMTP", to_address)
                    return
                except Exception as e:
                    logger.error("❌ SMTP email failed: %s. (Note: Render Free Tier blocks SMTP ports)", repr(e))
                    # Don't raise here if we want to just log the failure, 
                    # but raising helps debugging if it was the only method.
                    if not api_key:
                        raise e
            else:
                if not api_key:
                    logger.error("❌ No valid email configuration found (Resend or SMTP).")
                    raise RuntimeError("No valid email configuration found")

        except Exception as e:
            logger.error("❌ OTP email failed: %s", repr(e))

    threading.Thread(target=_worker, daemon=True).start()
    return True

def _client_ip_allowed() -> bool:
    if not config.ADMIN_IP_ALLOWLIST.strip():
        return True
    allowed = {ip.strip() for ip in config.ADMIN_IP_ALLOWLIST.split(",") if ip.strip()}
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "")
    client_ip = client_ip.split(",")[0].strip()
    return client_ip in allowed

def _require_admin():
    return bool(session.get('IsAdmin')) and _client_ip_allowed() and app.debug

# -------------------------
# Session defaults
# -------------------------

@app.before_request
def ensure_session_defaults():
    if 'IsAdmin' not in session:
        session['IsAdmin'] = False
    if 'User' not in session:
        session['User'] = None
    try:
        ensure_db_connection()
    except Exception as e:
        logger.warning("DB reconnect failed: %s", e)

# -------------------------
# Health/Ready
# -------------------------

@app.route('/health')
def health():
    return {"status": "ok", "time": time.time()}

@app.route('/ready')
def ready():
    db_ok = True
    try:
        ensure_db_connection()
        with mydb.cursor() as cur:
            cur.execute("SELECT 1")
    except Exception:
        db_ok = False
    model_ok = os.path.exists("Trained.yml")
    encoder_ok = os.path.exists("encoder.pkl")
    return {
        "db": db_ok,
        "model": model_ok,
        "encoder": encoder_ok,
        "has_cv2_face": has_cv2_face(),
    }, (200 if (db_ok and has_cv2_face()) else 503)

# -------------------------
# Routes
# -------------------------

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/admin', methods=['POST', 'GET'])
def admin():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        if (email == 'admin@voting.com') and (password == 'admin'):
            session['IsAdmin'] = True
            session['User'] = 'admin'
            flash('Admin login successful', 'success')
            logger.info("Admin logged in.")
        else:
            flash('Invalid admin credentials', 'danger')
    return render_template('admin.html', admin=session.get('IsAdmin', False))

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('home'))

@app.route('/add_nominee', methods=['POST', 'GET'])
def add_nominee():
    if request.method == 'POST':
        member = request.form.get('name', '').strip()
        party = request.form.get('party', '').strip()
        predefined_symbol = request.form.get('predefined_symbol', '')
        
        logo_filename = ""

        # Handle Image Logic
        if predefined_symbol and predefined_symbol != 'custom':
            logo_filename = predefined_symbol
        else:
            # Handle File Upload
            if 'image' not in request.files:
                flash('No file part', 'danger')
                return redirect(request.url)
            file = request.files['image']
            if file.filename == '':
                flash('No selected file', 'danger')
                return redirect(request.url)
            if file:
                filename = secure_filename(file.filename)
                # Ensure unique filename to prevent overwrites
                import uuid
                unique_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
                file.save(os.path.join('static/img', unique_filename))
                logo_filename = unique_filename

        nominee = read_sql('SELECT * FROM nominee', mydb)
        all_members = set(nominee.member_name.astype(str).str.strip().values) if not nominee.empty else set()
        all_parties = set(nominee.party_name.astype(str).str.strip().values) if not nominee.empty else set()
        
        if not member or not party or not logo_filename:
            flash('All fields are required', 'warning')
        elif member in all_members:
            flash('The member already exists', 'info')
        elif party in all_parties:
            flash('The party already exists', 'info')
        else:
            sql = "INSERT INTO nominee (member_name, party_name, symbol_name) VALUES (%s, %s, %s)"
            with mydb.cursor() as cur:
                cur.execute(sql, (member, party, logo_filename))
            mydb.commit()
            flash('Successfully registered a new nominee', 'success')
            logger.info("Nominee added: %s, %s, %s", member, party, logo_filename)
            
    return render_template('nominee.html', admin=session.get('IsAdmin', False))

import logging, datetime, random, time
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@app.route('/registration', methods=['POST', 'GET'])
def registration():
    if request.method == 'POST':
        first_name = request.form.get('first_name', '').strip()
        middle_name = request.form.get('middle_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        aadhar_id = request.form.get('aadhar_id', '').strip()
        voter_id = request.form.get('voter_id', '').strip()
        state = request.form.get('state', '').strip()
        d_name = request.form.get('d_name', '').strip()
        pno = request.form.get('pno', '').strip()
        email = request.form.get('email', '').strip()
        age_str = request.form.get('age', '').strip()

        try:
            age = int(age_str or "0")
            if age < 18 or age > 120:
                flash("Age must be between 18 and 120", "warning")
                return render_template('voter_reg.html')
            if not valid_email(email):
                flash("Please enter a valid email address", "warning")
                return render_template('voter_reg.html')
            if not valid_phone(pno):
                flash("Please enter a valid phone number", "warning")
                return render_template('voter_reg.html')
        except Exception as e:
            app.logger.exception("Validation error: %s", e)
            flash("Invalid input. Please check your details.", "danger")
            return render_template('voter_reg.html')

        try:
            with mydb.cursor() as cur:
                cur.execute("SELECT 1")
        except Exception as e:
            app.logger.exception("DB connectivity error: %s", e)
            flash("Database is unavailable right now. Please try again.", "danger")
            return render_template('voter_reg.html')

        try:
            with mydb.cursor() as cur:
                cur.execute("""CREATE TABLE IF NOT EXISTS pending_voters (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    first_name VARCHAR(50), middle_name VARCHAR(50), last_name VARCHAR(50),
                    aadhar_id VARCHAR(20), voter_id VARCHAR(20),
                    email VARCHAR(100), pno VARCHAR(15), state VARCHAR(50), d_name VARCHAR(50),
                    otp_code VARCHAR(10), otp_expires_at DATETIME,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
        except Exception as e:
            app.logger.exception("Ensuring pending_voters failed: %s", e)
            flash("Server setup error. Please try again later.", "danger")
            return render_template('voter_reg.html')

        try:
            with mydb.cursor() as cur:
                cur.execute("""CREATE TABLE IF NOT EXISTS voters (
                    sno INT AUTO_INCREMENT PRIMARY KEY,
                    first_name VARCHAR(100), middle_name VARCHAR(100), last_name VARCHAR(100),
                    aadhar_id VARCHAR(100), voter_id VARCHAR(100),
                    email VARCHAR(100), pno VARCHAR(100), state VARCHAR(100), d_name VARCHAR(100),
                    verified VARCHAR(100)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
        except Exception as e:
            app.logger.exception("Ensuring voters table failed: %s", e)
            flash("Server setup error (voters). Please try again later.", "danger")
            return render_template('voter_reg.html')

        try:
            with mydb.cursor() as cur:
                cur.execute("SELECT COUNT(*) as cnt FROM voters WHERE aadhar_id=%s OR voter_id=%s",
                            (aadhar_id, voter_id))
                res = cur.fetchone()
                exists = res['cnt'] > 0 if res else False
            if exists:
                flash("Already Registered as a Voter", "info")
                return render_template('voter_reg.html')
        except Exception as e:
            app.logger.exception("Duplicate check failed: %s", e)
            flash("We couldn't validate your record right now. Try again later.", "danger")
            return render_template('voter_reg.html')

        try:
            otp_code = str(random.randint(100000, 999999))
            expiry = datetime.datetime.now() + datetime.timedelta(minutes=5)
            with mydb.cursor() as cur:
                cur.execute("""
                    INSERT INTO pending_voters
                    (first_name, middle_name, last_name, aadhar_id, voter_id, email, pno, state, d_name, otp_code, otp_expires_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (first_name, middle_name, last_name, aadhar_id, voter_id, email, pno, state, d_name, otp_code, expiry))
            mydb.commit()
        except Exception as e:
            app.logger.exception("Insert pending_voters failed: %s", e)
            flash("Could not start verification. Please try again.", "danger")
            return render_template('voter_reg.html')

        try:
            if 'send_otp_email' in globals():
                send_otp_email(email, otp_code)
            elif 'send_email_otp' in globals():
                send_email_otp(email, otp_code)
            else:
                app.logger.warning("No OTP email function found; skipping send.")
        except Exception as e:
            app.logger.exception("OTP send failed: %s", e)
            flash("Failed to send OTP. Please try again later.", "danger")
            return render_template('voter_reg.html')

        session['voter_id'] = voter_id
        session['email'] = email
        session['status'] = 'no'
        session['otp_last_sent'] = int(time.time())

        flash("OTP sent! Please verify to complete registration.", "info")
        return render_template('verify.html')

    return render_template('voter_reg.html')
@app.route('/verify', methods=['POST', 'GET'])
def verify():
    try:
        if session.get('status') == 'no':
            if request.method == 'POST':
                otp_check = request.form.get('otp_check', '').strip()
                voter_id = session.get('voter_id')
                if not voter_id:
                    flash("Session expired. Please re-register.", "danger")
                    return redirect(url_for('registration'))
                with mydb.cursor(pymysql.cursors.DictCursor) as cur:
                    # Use Python time for consistency
                    now_time = datetime.datetime.now()
                    
                    # 1. Try Pending Voters (Registration)
                    cur.execute("""
                        SELECT * FROM pending_voters
                        WHERE voter_id=%s AND otp_code=%s AND otp_expires_at > %s
                    """, (voter_id, otp_check, now_time))
                    pending = cur.fetchone()
                    
                    if pending:
                        # Registration Success
                        cur.execute("""
                            INSERT INTO voters (first_name, middle_name, last_name, aadhar_id, voter_id,
                                                email, pno, state, d_name, verified)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """, (pending['first_name'], pending['middle_name'], pending['last_name'],
                                pending['aadhar_id'], pending['voter_id'], pending['email'],
                                pending['pno'], pending['state'], pending['d_name'], 'yes'))
                        cur.execute("DELETE FROM pending_voters WHERE id=%s", (pending['id'],))
                        mydb.commit()
                        
                        session['status'] = 'yes'
                        session['aadhar'] = pending['aadhar_id']
                        flash("Email verified successfully!", "success")
                        logger.info("Email verified (registration) for aadhar: %s", session.get('aadhar'))
                        
                    else:
                        # 2. Try Active Voters (Voting Login)
                        cur.execute("""
                            SELECT * FROM voters
                            WHERE voter_id=%s AND otp_code=%s AND otp_expires_at > %s
                        """, (voter_id, otp_check, now_time))
                        voter = cur.fetchone()
                        
                        if voter:
                            # Login Verification Success
                            # Clear OTP to prevent reuse
                            cur.execute("UPDATE voters SET otp_code=NULL, otp_expires_at=NULL WHERE voter_id=%s", (voter_id,))
                            mydb.commit()
                            
                            session['status'] = 'yes'
                            session['aadhar'] = voter['aadhar_id']
                            flash("Identity verified successfully!", "success")
                            logger.info("Identity verified (login) for aadhar: %s", session.get('aadhar'))
                        else:
                            # Failed both
                            logger.warning(f"OTP Fail for {voter_id}: Input={otp_check} (Checked Pending & Active)")
                            flash("Invalid or expired OTP. Please try again.", "warning")
                            return render_template('verify.html')

                next_step = session.pop('post_verify_next', None)
                if next_step == 'voting':
                    flash("Verified. Continue to voting.", "success")
                    return redirect(url_for('voting'))
                return redirect(url_for('capture_images'))
            else:
                # Resend Logic
                now = int(time.time())
                last_sent = session.get('otp_last_sent', 0)
                wait_min = getattr(config, 'OTP_RESEND_MIN_INTERVAL', 30)
                if now - last_sent < wait_min:
                    wait = wait_min - (now - last_sent)
                    flash(f"Please wait {wait}s before requesting another OTP.", "info")
                    return render_template('verify.html')
                
                receiver_address = session.get('email')
                voter_id = session.get('voter_id')
                
                if not receiver_address or not voter_id:
                    flash("Missing session data. Please re-register/login.", "danger")
                    return redirect(url_for('registration'))
                
                otp_code = str(random.randint(100000, 999999))
                expiry = datetime.datetime.now() + datetime.timedelta(minutes=5)
                
                try:
                    with mydb.cursor() as cur:
                        # Check where the user is
                        cur.execute("SELECT 1 FROM pending_voters WHERE voter_id=%s", (voter_id,))
                        is_pending = cur.fetchone()
                        
                        if is_pending:
                            cur.execute("""
                                UPDATE pending_voters
                                SET otp_code=%s, otp_expires_at=%s
                                WHERE voter_id=%s
                            """, (otp_code, expiry, voter_id))
                        else:
                            # Assume active voter
                            cur.execute("""
                                UPDATE voters
                                SET otp_code=%s, otp_expires_at=%s
                                WHERE voter_id=%s
                            """, (otp_code, expiry, voter_id))
                        mydb.commit()
                    
                    if 'send_otp_email' in globals():
                        send_otp_email(receiver_address, otp_code)
                    elif 'send_email_otp' in globals():
                        send_email_otp(receiver_address, otp_code)
                    
                    session['otp_last_sent'] = now
                    flash("OTP sent to your email.", "info")
                except Exception as e:
                    app.logger.error("Failed to resend OTP: %s", e)
                    flash("Failed to send OTP. Please try again later.", "danger")
                return render_template('verify.html')
        else:
            flash("Your email is already verified", "warning")
            return redirect(url_for('capture_images'))
    except pymysql.err.IntegrityError as dup:
        app.logger.warning("Duplicate during verify: %s", dup)
        flash("This voter already exists in the system.", "warning")
        return redirect(url_for('login'))
    except Exception as e:
        app.logger.error("Error in /verify route: %s", e)
        flash("Something went wrong during verification. Please try again later.", "danger")
        return redirect(url_for('registration'))
@app.route('/capture_images', methods=['POST', 'GET'])
def capture_images():
    # Guard: require logged-in aadhar and verified email before allowing capture
    if not session.get('aadhar'):
        flash("Please register/login first to capture images.", "warning")
        return redirect(url_for('registration'))
    if session.get('status') != 'yes':
        flash("Please verify your email before capturing images.", "warning")
        return redirect(url_for('verify'))

    # Helper function for face validation
    def face_centered_and_large(frame_w: int, frame_h: int, x: int, y: int, w: int, h: int) -> bool:
        min_dim = min(frame_w, frame_h)
        size_ok = (w >= config.CAPTURE_MIN_FACE_RATIO * min_dim) and (h >= config.CAPTURE_MIN_FACE_RATIO * min_dim)
        cx_face = x + w / 2.0
        cy_face = y + h / 2.0
        cx_img = frame_w / 2.0
        cy_img = frame_h / 2.0
        tol_x = config.CAPTURE_CENTER_TOLERANCE * frame_w
        tol_y = config.CAPTURE_CENTER_TOLERANCE * frame_h
        center_ok = (abs(cx_face - cx_img) <= tol_x) and (abs(cy_face - cy_img) <= tol_y)
        return size_ok and center_ok

    # Handle AJAX POST for image upload
    if request.method == 'POST' and request.is_json:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return {"status": "error", "reason": "No image data"}

        aadhar = session.get('aadhar')
        if not aadhar:
            return {"status": "error", "reason": "Session expired"}

        # Decode Base64 Image
        try:
            import base64
            header, encoded = image_data.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                return {"status": "error", "reason": "Invalid image"}
        except Exception as e:
            return {"status": "error", "reason": f"Decode failed: {str(e)}"}

        # Prepare storage path
        path_to_store = os.path.join(os.getcwd(), "all_images", aadhar)
        os.makedirs(path_to_store, exist_ok=True)
        
        # Check current count
        current_count = len([name for name in os.listdir(path_to_store) if name.endswith('.jpg')])
        if current_count >= 100:
            return {"status": "finished"}

        # Process Image (Face Detection)
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            frame_h, frame_w = gray.shape[:2]
            faces = cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                return {"status": "ignored", "reason": "no_face"}

            # Find the best face (largest)
            best_face = None
            max_area = 0
            
            for (x, y, w, h) in faces:
                if w * h > max_area:
                    max_area = w * h
                    best_face = (x, y, w, h)

            if best_face:
                x, y, w, h = best_face
                
                # Validation Checks
                if w < config.MIN_FACE_SIZE or h < config.MIN_FACE_SIZE:
                    return {"status": "ignored", "reason": "too_small"}
                
                roi = gray[y:y + h, x:x + w]
                if is_blurry(roi):
                    return {"status": "ignored", "reason": "blurry"}
                
                if not face_centered_and_large(frame_w, frame_h, x, y, w, h):
                    return {"status": "ignored", "reason": "not_centered"}

                # Save Image
                sampleNum = current_count + 1
                cv2.imwrite(os.path.join(path_to_store, f"{sampleNum}.jpg"), roi)
                return {"status": "success", "count": sampleNum}
            
            return {"status": "ignored", "reason": "no_valid_face"}

        except Exception as e:
            logger.error("Error processing frame: %s", e)
            return {"status": "error", "reason": "Processing error"}

    # GET request: Render the capture page
    return render_template('capture.html')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

def getImagesAndLabels(path: str) -> Tuple[List[np.ndarray], List[int]]:
    if not os.path.isdir(path):
        return [], []

    person_dirs = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    faces: List[np.ndarray] = []
    labels: List[str] = []

    for folder in person_dirs:
        aadhar_id = os.path.basename(folder)
        imagePaths = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        imagePaths = [p for p in imagePaths if os.path.splitext(p)[1].lower() in ('.jpg', '.jpeg', '.png')]
        for imagePath in imagePaths:
            try:
                pilImage = Image.open(imagePath).convert('L')
                imageNp = np.array(pilImage, 'uint8')
                if imageNp.shape[0] < config.MIN_FACE_SIZE or imageNp.shape[1] < config.MIN_FACE_SIZE:
                    continue
                if is_blurry(imageNp):
                    continue
                faces.append(imageNp)
                labels.append(aadhar_id)
            except Exception:
                continue

    if not labels:
        return [], []

    label_ids = le.fit_transform(labels).tolist()
    with open('encoder.pkl', 'wb') as output:
        pickle.dump(le, output)
    return faces, label_ids

@app.route('/train', methods=['POST', 'GET'])
def train():
    if request.method == 'POST':
        if not has_cv2_face():
            flash("OpenCV contrib (cv2.face) is not available. Install opencv-contrib-python.", "danger")
            return redirect(url_for('home'))

        faces, label_ids = getImagesAndLabels("all_images")
        if len(faces) == 0:
            flash("No images found to train. Please capture images first.", "warning")
            return redirect(url_for('home'))

        with open('encoder.pkl', 'rb') as f:
            enc = pickle.load(f)
        classes = list(getattr(enc, "classes_", []))
        if len(classes) < 2:
            flash("Training requires at least 2 different voters (labels).", "warning")
            return redirect(url_for('home'))

        counts = pd.Series(label_ids).value_counts()
        if (counts < config.MIN_IMAGES_PER_LABEL).any():
            flash(f"Each voter should have at least {config.MIN_IMAGES_PER_LABEL} good images. Please capture more.", "info")
            return redirect(url_for('home'))

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(label_ids))
        recognizer.save("Trained.yml")
        load_encoder(force=True)
        load_recognizer(force=True)

        flash("Model trained successfully.", 'success')
        logger.info("Model trained. Faces: %d, Labels: %d", len(faces), len(set(label_ids)))
        return redirect(url_for('home'))
    return render_template('train.html')

@app.route('/update')
def update():
    return render_template('update.html')

@app.route('/updateback', methods=['POST', 'GET'])
def updateback():
    if request.method == 'POST':
        first_name = request.form.get('first_name', '').strip()
        middle_name = request.form.get('middle_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        aadhar_id = request.form.get('aadhar_id', '').strip()
        voter_id = request.form.get('voter_id', '').strip()
        email = request.form.get('email', '').strip()
        pno = request.form.get('pno', '').strip()
        age_str = request.form.get('age', '').strip()

        try:
            age = int(age_str)
        except Exception:
            age = 0

        voters = read_sql('SELECT aadhar_id FROM voters', mydb)
        all_aadhar_ids = set(voters.aadhar_id.astype(str).str.strip().values) if not voters.empty else set()

        if age < 18 or age > 120:
            flash("Age must be between 18 and 120", "warning")
        elif aadhar_id in all_aadhar_ids:
            sql = ("UPDATE voters SET first_name=%s, middle_name=%s, last_name=%s, "
                   "voter_id=%s, email=%s, pno=%s, verified=%s WHERE aadhar_id=%s")
            with mydb.cursor() as cur:
                cur.execute(sql, (first_name, middle_name, last_name, voter_id, email, pno, 'no', aadhar_id))
            mydb.commit()
            session['aadhar'] = aadhar_id
            session['status'] = 'yes'
            session['email'] = email
            flash('Database updated successfully. Please re-verify email.', 'success')
            logger.info("Voter updated: %s", aadhar_id)
            return redirect(url_for('verify'))
        else:
            flash(f"Aadhar: {aadhar_id} doesn't exist in the database for update", 'warning')
    return render_template('update.html')

# NEW: Login with details (voter_id + aadhar_id + email) for voting
@app.route('/login_details', methods=['GET', 'POST'])
def login_details():
    # Clear session on GET to ensure fresh login
    if request.method == 'GET':
        session.pop('aadhar', None)
        session.pop('email', None)
        session.pop('status', None)
        session.pop('vote', None)
        session.pop('select_aadhar', None)

    if request.method == 'POST':
        voter_id = request.form.get('voter_id', '').strip()
        aadhar = request.form.get('aadhar_id', '').strip()
        email = request.form.get('email', '').strip()

        if not voter_id or not aadhar or not email:
            flash("All fields are required.", "warning")
            return render_template('login_details.html')

        try:
            # 1. Check if user exists by Aadhar first
            df_aadhar = read_sql('SELECT * FROM voters WHERE aadhar_id=%s LIMIT 1', mydb, params=[aadhar])
            
            if df_aadhar.empty:
                flash("User not found. Please register first.", "danger")
                return render_template('login_details.html')
            
            # 2. Check if details match
            user_row = df_aadhar.iloc[0]
            db_voter_id = str(user_row['voter_id']).strip()
            db_email = str(user_row['email']).strip()
            
            if (db_voter_id != voter_id) or (db_email != email):
                flash("Details do not match. Please check your Voter ID and Email.", "warning")
                return render_template('login_details.html')

            # 3. Check verification status - FORCE RE-VERIFICATION (2FA)
            # verified = str(user_row['verified'] or '').lower()
            # session['status'] = 'yes' if verified == 'yes' else 'no'
            
            # Set session
            session['aadhar'] = str(user_row['aadhar_id'])
            session['email'] = db_email
            session['voter_id'] = db_voter_id # Ensure voter_id is in session for verify
            
            # Always force 'no' to require OTP on every login
            session['status'] = 'no'
            session['post_verify_next'] = 'voting'

            # 4. Check if already voted
            try:
                df_exists = read_sql('SELECT 1 FROM vote WHERE aadhar=%s LIMIT 1', mydb, params=[session['aadhar']])
                if not df_exists.empty:
                    return redirect(url_for('already_voted'))
            except Exception as e:
                logger.warning("Login duplicate check failed: %s", e)

            # Generate and Send OTP for Voting Login
            otp_code = str(random.randint(100000, 999999))
            expiry = datetime.datetime.now() + datetime.timedelta(minutes=5)
            
            try:
                with mydb.cursor() as cur:
                    cur.execute("""
                        UPDATE voters 
                        SET otp_code=%s, otp_expires_at=%s 
                        WHERE voter_id=%s
                    """, (otp_code, expiry, db_voter_id))
                
                if 'send_otp_email' in globals():
                    send_otp_email(db_email, otp_code)
                elif 'send_email_otp' in globals():
                    send_email_otp(db_email, otp_code)
                
                session['otp_last_sent'] = int(time.time())
                flash("Details matched. OTP sent to your email.", "info")
            except Exception as e:
                logger.error("Failed to generate/send voting OTP: %s", e)
                flash("Error sending OTP. Please try again.", "danger")
                return render_template('login_details.html')

            return redirect(url_for('verify'))

        except Exception as e:
            logger.warning("Login by details failed: %s", e)
            flash("Error while verifying details. Try again.", "danger")
            return render_template('login_details.html')

    return render_template('login_details.html')

# Existing simple login by aadhar (kept for admin/testing if needed)
@app.route('/login_aadhar', methods=['GET', 'POST'])
def login_aadhar():
    if request.method == 'POST':
        aadhar = request.form.get('aadhar', '').strip()
        if not aadhar:
            flash("Please enter your Aadhar ID.", "warning")
            return redirect(url_for('login_aadhar'))
        try:
            df = read_sql('SELECT aadhar_id, email, verified FROM voters WHERE aadhar_id=%s LIMIT 1', mydb, params=[aadhar])
            if df.empty:
                flash("Aadhar not found. Please register first.", "warning")
                return redirect(url_for('login_aadhar'))
            # Set session from DB
            session['aadhar'] = str(df.iloc[0]['aadhar_id'])
            session['email'] = str(df.iloc[0]['email'])
            verified = str(df.iloc[0]['verified'] or '').lower()
            session['status'] = 'yes' if verified == 'yes' else 'no'

            # If not verified, send to OTP first, then return to voting
            if session['status'] != 'yes':
                session['post_verify_next'] = 'voting'
                flash("Please verify your email to continue to voting.", "info")
                return redirect(url_for('verify'))

            flash("Logged in with Aadhar successfully.", "success")
            return redirect(url_for('voting'))
        except Exception as e:
            logger.warning("Login by aadhar failed: %s", e)
            flash("Error while logging in. Try again.", "danger")
            return redirect(url_for('login_aadhar'))
    # Simple inline form (no template needed)
    return """
    <h2>Login by Aadhar</h2>
    <form method="post">
      <label>Aadhar ID:</label>
      <input type="text" name="aadhar" />
      <button type="submit">Login</button>
    </form>
    """

@app.route('/voting', methods=['POST', 'GET'])
def voting():
    # If user lands here via GET and has no aadhar, guide them to the details form
    if request.method == 'GET' and not session.get('aadhar'):
        flash("Please enter your Voter ID, Aadhar ID, and Email to continue.", "info")
        return redirect(url_for('login_details'))

    if request.method == 'POST':
        if not has_cv2_face():
            flash("OpenCV contrib (cv2.face) is not available. Install opencv-contrib-python.", "danger")
            return render_template('voting.html')

        expected_aadhar = session.get('aadhar')
        if not expected_aadhar:
            flash("Please enter your Voter ID, Aadhar ID, and Email to continue.", "info")
            return redirect(url_for('login_details'))

        # If email not verified, send to verify and return here after success
        if session.get('status') != 'yes':
            session['post_verify_next'] = 'voting'
            flash("Please verify your email to continue to voting.", "info")
            return redirect(url_for('verify'))

        try:
            df_exists = read_sql('SELECT 1 FROM vote WHERE aadhar=%s LIMIT 1', mydb, params=[expected_aadhar])
            if not df_exists.empty:
                # User already voted. Redirect to already_voted page.
                return redirect(url_for('already_voted'))
        except Exception as e:
            logger.warning("Pre-voting duplicate check failed: %s", e)

        encoder = load_encoder()
        recognizer = load_recognizer()
        if encoder is None or recognizer is None:
            flash("Model not trained yet. Please train the model first.", "warning")
            return render_template('voting.html')

        try:
            classes = list(getattr(encoder, "classes_", []))
        except Exception:
            classes = []
        if expected_aadhar not in classes:
            flash("Your face is not in the trained model. Please capture your images and retrain before voting.", "warning")
            return render_template('voting.html')

        expected_label_id = int(np.where(np.array(classes) == expected_aadhar)[0][0])

        cam = try_open_camera()
        if not cam:
            flash("Unable to access camera. Check permissions and device index.", "warning")
            return render_template('voting.html')

        det_aadhar = None
        start_time = time.time()
        frames_seen = 0
        consec_ok = 0

        try:
            while True:
                ret, im = cam.read()
                if not ret:
                    continue

                frames_seen += 1
                if (time.time() - start_time) > config.VOTING_MAX_SECONDS or frames_seen >= config.VOTING_MAX_FRAMES:
                    break

                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=config.VOTING_SCALE_FACTOR,
                    minNeighbors=config.VOTING_MIN_NEIGHBORS
                )
                current_frame_ok = False
                for (x, y, w, h) in faces:
                    if w < config.VOTING_MIN_FACE_SIZE or h < config.VOTING_MIN_FACE_SIZE:
                        continue
                    try:
                        roi = gray[y:y + h, x:x + w]
                        roi_resized = cv2.resize(roi, (config.ROI_SIZE, config.ROI_SIZE))
                        pred_id, conf = recognizer.predict(roi_resized)
                        if app.debug:
                            logger.info("Voting strict: conf=%.2f, pred=%s exp=%s", conf, str(pred_id), str(expected_label_id))
                    except cv2.error as e:
                        if app.debug:
                            logger.warning("predict error: %s", e)
                        record_attempt(None, False, None)
                        continue

                    if pred_id == expected_label_id and conf < config.VOTING_STRICT_CONF_THRESHOLD:
                        current_frame_ok = True
                        break

                if current_frame_ok:
                    consec_ok += 1
                else:
                    consec_ok = 0

                record_attempt(conf if 'conf' in locals() else None, current_frame_ok, expected_label_id if current_frame_ok else None)

                if consec_ok >= config.VOTING_REQUIRED_CONSEC_MATCHES:
                    det_aadhar = expected_aadhar
                    break
        finally:
            cam.release()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        if det_aadhar:
            # Post-recognition duplicate-vote check (A3): if already voted, show message and stop
            try:
                df_exists = read_sql('SELECT 1 FROM vote WHERE aadhar=%s LIMIT 1', mydb, params=[det_aadhar])
                if not df_exists.empty:
                    flash("You already voted", "warning")
                    return render_template('voting.html')
            except Exception as e:
                logger.warning("Post-recognition duplicate-vote check failed: %s", e)
                # If the DB check fails, we proceed to candidate selection as before.

            session['select_aadhar'] = det_aadhar
            logger.info("Detected voter (strict): %s", det_aadhar)
            return redirect(url_for('select_candidate'))
        else:
            if _require_admin():
                flash("Unable to verify the expected voter. Admin can override below (debug only).", "warning")
                session['override_allowed'] = True
                return render_template('voting.html', admin_override=True)
            flash("Unable to verify your identity for voting. Please contact help desk.", "info")
            return render_template('voting.html')
    return render_template('voting.html')

@app.route('/voting/override', methods=['POST'])
def voting_override():
    if not _require_admin():
        return "Unauthorized", 403
    if not session.get('override_allowed'):
        flash("No override context. Start voting first.", "warning")
        return redirect(url_for('voting'))
    aadhar = request.form.get('aadhar', '').strip()
    if not aadhar:
        flash("Aadhar is required to override.", "warning")
        return redirect(url_for('voting'))
    try:
        df = read_sql('SELECT aadhar_id FROM voters WHERE aadhar_id=%s', mydb, params=[aadhar])
        if df.empty:
            flash("Aadhar not found in voters.", "danger")
            return redirect(url_for('voting'))
    except Exception as e:
        logger.warning("DB check error: %s", e)
        flash("Error verifying aadhar.", "danger")
        return redirect(url_for('voting'))
    session['select_aadhar'] = aadhar
    session.pop('override_allowed', None)
    flash("Admin override accepted. Proceed to candidate selection.", "info")
    return redirect(url_for('select_candidate'))

@app.route('/select_candidate', methods=['POST', 'GET'])
def select_candidate():
    aadhar = session.get('select_aadhar')
    if not aadhar:
        flash("No detected voter found. Please try voting again.", "warning")
        return redirect(url_for('voting'))

    df_nom = read_sql('SELECT * FROM nominee', mydb)
    # We need full rows for the template (result) AND symbol names for validation (all_nom)
    all_nom = df_nom['symbol_name'].values if not df_nom.empty else []
    
    if df_nom.empty:
        flash("No nominees available yet. Please contact admin.", "info")
        return redirect(url_for('home'))

    g = read_sql('SELECT aadhar FROM vote', mydb)
    all_adhar = g['aadhar'].values if not g.empty else []
    if aadhar in all_adhar:
        flash("You already voted", "warning")
        return redirect(url_for('home'))
    else:
        if request.method == 'POST':
            # Template sends 'nominee'
            vote = request.form.get('nominee')

            # Defensive, authoritative check immediately before insert
            try:
                df_exists = read_sql('SELECT 1 FROM vote WHERE aadhar=%s LIMIT 1', mydb, params=[aadhar])
                if not df_exists.empty:
                    flash("You already voted", "warning")
                    logger.info("Duplicate vote prevented for aadhar: %s", aadhar)
                    return redirect(url_for('home'))
            except Exception as e:
                logger.warning("Pre-insert duplicate check failed: %s", e)
                # We continue; DB may still enforce uniqueness if configured.

            if vote not in all_nom:
                flash("Invalid candidate selected.", "danger")
                return render_template('select_candidate.html', result=df_nom.values)
            
            # Look up the ID for the selected symbol (Fix for FK constraint)
            try:
                df_selected = read_sql('SELECT id FROM nominee WHERE symbol_name=%s', mydb, params=[vote])
                if df_selected.empty:
                    flash("Invalid candidate selected (ID not found).", "danger")
                    return render_template('select_candidate.html', result=df_nom.values)
                
                vote_id = int(df_selected.iloc[0]['id'])
                
                session['vote'] = vote # Keep symbol in session for display if needed
                
                # Insert the ID
                sql = "INSERT INTO vote (vote, aadhar) VALUES (%s, %s)"
                with mydb.cursor() as cur:
                    cur.execute(sql, (vote_id, aadhar))
                mydb.commit()
            except Exception as e:
                logger.error("Vote insertion failed: %s", e)
                flash("Error recording vote. Please try again.", "danger")
                return redirect(url_for('home'))
            except pymysql.err.IntegrityError as e:
                # In case a unique index exists on aadhar, catch and inform the user.
                logger.info("IntegrityError on vote insert for aadhar=%s: %s", aadhar, e)
                flash("You already voted", "warning")
                return redirect(url_for('home'))

            if config.FAST2SMS_API_KEY:
                try:
                    s = "SELECT pno, first_name FROM voters WHERE aadhar_id=%s"
                    c = read_sql(s, mydb, params=[aadhar])
                    pno = str(c.values[0][0])
                    name = str(c.values[0][1])
                    url = "https://www.fast2sms.com/dev/bulkV2"
                    message = f"Dear {name}, your vote has been recorded. Thank you."
                    data1 = {
                        "route": "q",
                        "message": message,
                        "language": "english",
                        "flash": 0,
                        "numbers": pno,
                    }
                    headers = {
                        "authorization": config.FAST2SMS_API_KEY,
                        "Content-Type": "application/json"
                    }
                    import requests
                    requests.post(url, headers=headers, json=data1, timeout=10)
                except Exception as e:
                    logger.warning("SMS send error: %s", e)

            flash("Voted Successfully", 'success')
            logger.info("Vote recorded for aadhar: %s, vote: %s", aadhar, vote)
            
            # CLEAR SESSION so the next user must login
            session.pop('aadhar', None)
            session.pop('email', None)
            session.pop('status', None)
            session.pop('vote', None)
            session.pop('select_aadhar', None)
            
            return redirect(url_for('home'))
    
    # Pass df_nom.values as 'result' for the template loop
    return render_template('select_candidate.html', result=df_nom.values)

@app.route('/voting_res')
def voting_res():
    # Get all nominees
    df_nom = read_sql('SELECT * FROM nominee', mydb)
    
    # Get all votes
    votes_df = read_sql('SELECT * FROM vote', mydb)
    
    result = []
    if not df_nom.empty:
        # Calculate vote counts
        vote_counts = {}
        if not votes_df.empty:
            vote_counts = votes_df['vote'].value_counts().to_dict()
            
        # Construct result list: [id, member_name, party_name, symbol_name, count]
        # Iterate over df_nom rows
        for index, row in df_nom.iterrows():
            symbol = str(row['symbol_name'])
            # Use ID to get count (Fix for schema change)
            count = vote_counts.get(row['id'], 0)
            # Create a list matching the template's expected indices
            # row[0]=id (unused in template), row[1]=name, row[2]=party, row[3]=symbol, row[4]=count
            entry = [
                row['id'],
                row['member_name'],
                row['party_name'],
                symbol,
                count
            ]
            result.append(entry)
            
    # Aggregate votes by party
    party_data = {}
    for entry in result:
        party = entry[2] # party_name is at index 2
        count = entry[4] # count is at index 4
        party_data[party] = party_data.get(party, 0) + count
    
    # Convert to list and sort by votes descending
    party_result = [[p, c] for p, c in party_data.items()]
    party_result.sort(key=lambda x: x[1], reverse=True)
            
    return render_template('voting_res.html', result=result, party_result=party_result)

@app.route('/already_voted')
def already_voted():
    return render_template('already_voted.html')

# -------------------------
# Admin-protected Debug/Inspection Routes
# -------------------------

@app.route('/admin/summary')
def admin_summary():
    if not _require_admin():
        return "Unauthorized", 403
    voters = read_sql('SELECT COUNT(*) AS total, SUM(verified="yes") AS verified FROM voters', mydb)
    nominees = read_sql('SELECT COUNT(*) AS total FROM nominee', mydb)
    votes = read_sql('SELECT COUNT(*) AS total FROM vote', mydb)
    voters_total = int(voters.iloc[0]['total'] or 0)
    voters_verified = int(voters.iloc[0]['verified'] or 0)
    nominees_total = int(nominees.iloc[0]['total'] or 0)
    votes_total = int(votes.iloc[0]['total'] or 0)
    html = f"""
    <h2>Admin Summary</h2>
    <ul>
      <li>Total voters: {voters_total}</li>
      <li>Verified voters: {voters_verified}</li>
      <li>Total nominees: {nominees_total}</li>
      <li>Total votes: {votes_total}</li>
    </ul>
    """
    return html

@app.route('/debug/db/voters')
def debug_db_voters():
    if not _require_admin():
        return "Unauthorized", 403
    df = read_sql('SELECT * FROM voters ORDER BY aadhar_id DESC LIMIT 100', mydb)
    return df.to_html(index=False)

@app.route('/debug/db/nominees')
def debug_db_nominees():
    if not _require_admin():
        return "Unauthorized", 403
    df = read_sql('SELECT * FROM nominee ORDER BY member_name ASC LIMIT 100', mydb)
    return df.to_html(index=False)

@app.route('/debug/db/votes')
def debug_db_votes():
    if not _require_admin():
        return "Unauthorized", 403
    df = read_sql('SELECT * FROM vote ORDER BY id DESC LIMIT 100', mydb)
    return df.to_html(index=False)

@app.route('/debug/session')
def debug_session():
    if not _require_admin():
        return "Unauthorized", 403
    keys = ['IsAdmin','User','aadhar','status','email','select_aadhar','vote','otp','otp_expiry','otp_last_sent','override_allowed']
    data = {k: session.get(k) for k in keys}
    rows = "".join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in data.items()])
    return f"<h2>Session</h2><table border='1'><tr><th>Key</th><th>Value</th></tr>{rows}</table>"

@app.route('/debug/images/<aadhar>')
def debug_images(aadhar):
    if not _require_admin():
        return "Unauthorized", 403
    path = os.path.join(os.getcwd(), "all_images", aadhar)
    if not os.path.isdir(path):
        return f"No folder for {aadhar}", 404
    files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files_sorted = sorted(files)
    preview = "<br>".join(files_sorted[:200])
    return f"<h2>Images for {aadhar}</h2><p>Total: {len(files_sorted)}</p><pre>{preview}</pre>"

@app.route('/debug/encoder')
def debug_encoder():
    if not _require_admin():
        return "Unauthorized", 403
    try:
        enc = load_encoder()
        if enc is None:
            return "<h2>Encoder</h2><p>encoder.pkl not found.</p>", 404
        classes = list(getattr(enc, 'classes_', []))
        return f"<h2>Encoder</h2><p>Classes ({len(classes)}):</p><pre>{classes}</pre>"
    except Exception as e:
        return f"<h2>Encoder</h2><p>Error: {e}</p>", 500

@app.route('/debug/model')
def debug_model():
    if not _require_admin():
        return "Unauthorized", 403
    exists = os.path.exists("Trained.yml")
    size = os.path.getsize("Trained.yml") if exists else 0
    return f"<h2>Model</h2><p>Exists: {exists}</p><p>Size: {size} bytes</p>"

@app.route('/debug/env')
def debug_env():
    if not _require_admin():
        return "Unauthorized", 403
    info = {
        "cwd": os.getcwd(),
        "opencv_version": getattr(cv2, '__version__', 'unknown'),
        "has_cv2_face": has_cv2_face(),
        "has_haarcascade": os.path.exists(facedata),
        "lbph_conf_threshold": config.LBPH_CONF_THRESHOLD,
        "voting_scale_factor": config.VOTING_SCALE_FACTOR,
        "voting_min_neighbors": config.VOTING_MIN_NEIGHBORS,
        "voting_min_face_size": config.VOTING_MIN_FACE_SIZE,
        "roi_size": config.ROI_SIZE,
        "voting_strict_conf_threshold": config.VOTING_STRICT_CONF_THRESHOLD,
        "voting_required_consec_matches": config.VOTING_REQUIRED_CONSEC_MATCHES,
    }
    rows = "".join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in info.items()])
    return f"<h2>Environment</h2><table border='1'><tr><th>Key</th><th>Value</th></tr>{rows}</table>"

# -------------------------
# Admin Stats and Export/Import
# -------------------------

@app.route('/admin/stats')
def admin_stats():
    if not _require_admin():
        return "Unauthorized", 403
    base = os.path.join(os.getcwd(), "all_images")
    per_class_counts = {}
    if os.path.isdir(base):
        for name in sorted(os.listdir(base)):
            p = os.path.join(base, name)
            if os.path.isdir(p):
                cnt = sum(1 for f in os.listdir(p) if f.lower().endswith(('.jpg','.jpeg','.png')))
                per_class_counts[name] = cnt

    sum_attempts = summarize_attempts()
    rows = "".join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in sum_attempts.items() if k != "per_label_counts"])
    per_label_rows = "".join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in sum_attempts.get("per_label_counts", {}).items()])
    per_class_rows = "".join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in per_class_counts.items()])

    html = f"""
    <h2>Admin Stats</h2>
    <h3>Recent Recognition Attempts (last {RECENT_ATTEMPTS.maxlen})</h3>
    <table border='1'>
      <tr><th>Metric</th><th>Value</th></tr>
      {rows}
    </table>
    <h3>Attempts Per Predicted Label</h3>
    <table border='1'>
      <tr><th>Label</th><th>Count</th></tr>
      {per_label_rows}
    </table>
    <h3>Training Samples Per Class</h3>
    <table border='1'>
      <tr><th>Aadhar</th><th>Images</th></tr>
      {per_class_rows}
    </table>
    """
    return html

@app.route('/admin/export')
def admin_export():
    if not _require_admin():
        return "Unauthorized", 403
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode='w', compression=zipfile.ZIP_DEFLATED) as z:
        for fname in ["encoder.pkl", "Trained.yml"]:
            if os.path.exists(fname):
                z.write(fname, arcname=fname)
        base = "all_images"
        if os.path.isdir(base):
            for root, _, files in os.walk(base):
                for f in files:
                    path = os.path.join(root, f)
                    arc = os.path.relpath(path, ".")
                    z.write(path, arcname=arc)
    mem.seek(0)
    return send_file(mem, mimetype='application/zip', as_attachment=True, download_name='voting_export.zip')

@app.route('/admin/import', methods=['POST'])
def admin_import():
    if not _require_admin():
        return "Unauthorized", 403
    if 'file' not in request.files:
        flash("No file uploaded.", "warning")
        return redirect(url_for('admin_summary'))
    f = request.files['file']
    if not f.filename.lower().endswith(".zip"):
        flash("Please upload a .zip file.", "warning")
        return redirect(url_for('admin_summary'))
    tmpdir = tempfile.mkdtemp(prefix="import_")
    zpath = os.path.join(tmpdir, "import.zip")
    f.save(zpath)
    with zipfile.ZipFile(zpath, 'r') as z:
        z.extractall(tmpdir)
    for fname in ["encoder.pkl", "Trained.yml"]:
        src = os.path.join(tmpdir, fname)
        if os.path.exists(src):
            shutil.move(src, os.path.join(os.getcwd(), fname))
    src_images = os.path.join(tmpdir, "all_images")
    dest_images = os.path.join(os.getcwd(), "all_images")
    if os.path.isdir(src_images):
        if os.path.isdir(dest_images):
            shutil.rmtree(dest_images)
        shutil.move(src_images, dest_images)
    shutil.rmtree(tmpdir, ignore_errors=True)
    load_encoder(force=True)
    load_recognizer(force=True)
    flash("Import completed.", "success")
    return redirect(url_for('admin_summary'))

# -------------------------
# Run
# -------------------------

if __name__ == '__main__':
    app.debug = True
    try:
        _require_secrets_in_production()
    except Exception as e:
        logger.error("Startup secret check failed: %s", e)
        raise

    if config.SECRET_KEY == "dev-secret-key-change-me":
        logger.warning("Using default SECRET_KEY. Set SECRET_KEY env var for production.")
    if config.DB_PASS == "Likhith2411":
        logger.warning("Using default DB_PASS. Set DB_* env vars for production.")
    if config.MAIL_USER == "likhithreddygg@gmail.com":
        logger.warning("Configure MAIL_USER/MAIL_PASS for OTP emails.")

    db_ok = True
    try:
        ensure_db_connection()
        with mydb.cursor() as cur:
            cur.execute("SELECT 1")
    except Exception as e:
        db_ok = False
        logger.error("Database connectivity check failed: %s", e)
        if not app.debug:
            raise

    if not has_cv2_face():
        logger.error("OpenCV contrib (cv2.face) not available. Install opencv-contrib-python.")
        if not app.debug:
            raise SystemExit(1)

    app.run(host="0.0.0.0", port=int(__import__("os").environ.get("PORT", 5000)))





