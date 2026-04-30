from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timedelta, timezone
from typing import Optional, Any
import os
import sqlite3
import hashlib
import hmac
import secrets
import requests
import fal_client
import jwt
import time
import stripe

CREDIT_PACKS = {
    "30": {
        "price": 500,
        "credits": 30,
        "label": "Starter",
    },
    "100": {
        "price": 1500,
        "credits": 100,
        "label": "Best Seller",
    },
    "300": {
        "price": 4000,
        "credits": 300,
        "label": "Premium",
    },
}

# =========================================================
# CONFIG
# =========================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = BASE_DIR / "uploads"
DB_PATH = Path("/data/face_aging.db")

OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

FAL_KEY = os.getenv("FAL_KEY", "").strip()
JWT_SECRET = os.getenv("JWT_SECRET", "").strip()
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("ACCESS_TOKEN_EXPIRE_HOURS", "24"))

DEFAULT_FREE_CREDITS = int(os.getenv("DEFAULT_FREE_CREDITS", "0"))
MAX_AGE = int(os.getenv("MAX_AGE", "100"))
MODEL_ID = os.getenv("MODEL_ID", "fal-ai/image-apps-v2/age-modify")

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "").strip()
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()
STRIPE_SUCCESS_URL = os.getenv(
    "STRIPE_SUCCESS_URL",
    "http://localhost:1420/?session_id={CHECKOUT_SESSION_ID}",
)
STRIPE_CANCEL_URL = os.getenv("STRIPE_CANCEL_URL", "http://localhost:1420")
STRIPE_PACK_NAME = os.getenv("STRIPE_PACK_NAME", "10 credits Face Aging")
STRIPE_PACK_CREDITS = int(os.getenv("STRIPE_PACK_CREDITS", "10"))
STRIPE_PACK_PRICE_EUR_CENTS = int(os.getenv("STRIPE_PACK_PRICE_EUR_CENTS", "1000"))

DEV_ADMIN_SECRET = os.getenv("DEV_ADMIN_SECRET", "").strip()

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

CORS_ORIGINS_RAW = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:1420,http://127.0.0.1:1420,http://tauri.localhost,https://tauri.localhost"
)
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS_RAW.split(",") if origin.strip()]

ALLOWED_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

if not FAL_KEY:
    raise RuntimeError("FAL_KEY manquant dans le fichier .env")

if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET manquant dans le fichier .env")

if not STRIPE_SECRET_KEY:
    raise RuntimeError("STRIPE_SECRET_KEY manquant dans le fichier .env")

stripe.api_key = STRIPE_SECRET_KEY

# =========================================================
# APP
# =========================================================

app = FastAPI(title="Face Aging API PRO")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "stripe-signature", "x-dev-admin-secret"],
)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

security = HTTPBearer()

RATE_LIMIT_STORE: dict[str, list[float]] = {}
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "10"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

# =========================================================
# DATABASE
# =========================================================

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            credits INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS generation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            requested_age INTEGER NOT NULL,
            credits_used INTEGER NOT NULL,
            output_filename TEXT,
            created_at TEXT NOT NULL,
            ip_address TEXT,
            status TEXT NOT NULL,
            error_message TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS credit_transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            transaction_type TEXT NOT NULL,
            amount INTEGER NOT NULL,
            balance_after INTEGER NOT NULL,
            stripe_payment_id TEXT,
            note TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_credit_transactions_stripe_payment_id
        ON credit_transactions(stripe_payment_id)
        WHERE stripe_payment_id IS NOT NULL
    """)

    conn.commit()
    conn.close()

@app.on_event("startup")
def startup():
    init_db()

# =========================================================
# SECURITY / AUTH
# =========================================================

def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000
    ).hex()
    return f"{salt}${pwd_hash}"

def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, pwd_hash = stored_hash.split("$", 1)
    except ValueError:
        return False

    new_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000
    ).hex()

    return hmac.compare_digest(new_hash, pwd_hash)

def create_access_token(user_id: int, email: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "email": email,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expiré")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token invalide")

def get_user_by_email(email: str) -> Optional[sqlite3.Row]:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),))
    user = cur.fetchone()
    conn.close()
    return user

def get_user_by_id(user_id: int) -> Optional[sqlite3.Row]:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cur.fetchone()
    conn.close()
    return user

def create_user(email: str, password: str) -> sqlite3.Row:
    conn = get_db()
    cur = conn.cursor()
    created_at = datetime.now(timezone.utc).isoformat()
    password_hash = hash_password(password)

    cur.execute(
        """
        INSERT INTO users (email, password_hash, credits, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (email.lower().strip(), password_hash, DEFAULT_FREE_CREDITS, created_at)
    )
    conn.commit()

    user_id = cur.lastrowid
    cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cur.fetchone()
    conn.close()
    return user

def update_user_credits(user_id: int, new_credits: int):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("UPDATE users SET credits = ? WHERE id = ?", (new_credits, user_id))
    conn.commit()
    conn.close()

def get_credit_transaction_by_stripe_payment_id(stripe_payment_id: str) -> Optional[sqlite3.Row]:
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM credit_transactions WHERE stripe_payment_id = ? LIMIT 1",
        (stripe_payment_id,)
    )
    row = cur.fetchone()
    conn.close()
    return row

def log_generation(
    user_id: int,
    requested_age: int,
    credits_used: int,
    output_filename: Optional[str],
    ip_address: Optional[str],
    status: str,
    error_message: Optional[str] = None,
):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO generation_logs
        (user_id, requested_age, credits_used, output_filename, created_at, ip_address, status, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            requested_age,
            credits_used,
            output_filename,
            datetime.now(timezone.utc).isoformat(),
            ip_address,
            status,
            error_message,
        )
    )
    conn.commit()
    conn.close()

def add_credit_transaction(
    user_id: int,
    transaction_type: str,
    amount: int,
    balance_after: int,
    stripe_payment_id: Optional[str] = None,
    note: Optional[str] = None,
):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO credit_transactions
        (user_id, transaction_type, amount, balance_after, stripe_payment_id, note, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            transaction_type,
            amount,
            balance_after,
            stripe_payment_id,
            note,
            datetime.now(timezone.utc).isoformat(),
        )
    )
    conn.commit()
    conn.close()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> sqlite3.Row:
    token = credentials.credentials
    payload = decode_access_token(token)
    user_id = int(payload["sub"])
    user = get_user_by_id(user_id)

    if not user:
        raise HTTPException(status_code=401, detail="Utilisateur introuvable")

    return user

def require_dev_admin(request: Request):
    if not DEV_ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Route désactivée")
    header_value = request.headers.get("x-dev-admin-secret", "").strip()
    if not header_value or not secrets.compare_digest(header_value, DEV_ADMIN_SECRET):
        raise HTTPException(status_code=403, detail="Accès refusé")

# =========================================================
# HELPERS
# =========================================================

def check_rate_limit(key: str):
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    timestamps = RATE_LIMIT_STORE.get(key, [])
    timestamps = [ts for ts in timestamps if ts > window_start]

    if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Trop de requêtes. Réessaie plus tard."
        )

    timestamps.append(now)
    RATE_LIMIT_STORE[key] = timestamps

def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"

def validate_uploaded_image(file: UploadFile, content: bytes):
    if not content:
        raise HTTPException(status_code=400, detail="Fichier vide")

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Fichier trop volumineux. Maximum {MAX_UPLOAD_MB} MB"
        )

    is_jpeg = content.startswith(b"\xff\xd8\xff")
    is_png = content.startswith(b"\x89PNG\r\n\x1a\n")
    is_webp = content.startswith(b"RIFF") and b"WEBP" in content[:20]

    if not (is_jpeg or is_png or is_webp):
        raise HTTPException(
            status_code=400,
            detail="Image invalide. Formats acceptés: jpg, jpeg, png, webp"
        )

# =========================================================
# ROUTES - PUBLIC
# =========================================================

@app.get("/")
def root():
    return {
        "success": True,
        "message": "Face Aging API PRO is running"
    }
@app.get("/payment-success")
def payment_success(session_id: str = ""):
    message = "Payment successful. You can now return to the app."

    try:
        clean_session_id = normalize_checkout_session_id(session_id)

        if clean_session_id:
            result = credit_paid_checkout_session(clean_session_id)
            print("PAYMENT SUCCESS CREDIT RESULT:", result)

            credits_added = int(result.get("credits_added", 0))
            credits_total = int(result.get("credits_total", 0))

            if credits_added > 0:
                message = f"Payment successful. {credits_added} credits added. Total: {credits_total}."
            else:
                message = f"Payment confirmed. Total credits: {credits_total}."
    except Exception as e:
        print("PAYMENT SUCCESS ERROR:", e)
        message = "Payment received, but automatic credit sync failed. Please contact support if credits do not appear."

    return HTMLResponse(f"""
    <html>
      <head><title>Payment successful</title></head>
      <body style="font-family:Arial;padding:40px;text-align:center;background:#0b1020;color:white;">
        <h1>Payment successful</h1>
        <p>{message}</p>
        <p>You can now return to the app.</p>
      </body>
    </html>
    """)
@app.get("/payment-cancel")
def payment_cancel():
    return HTMLResponse("""
    <html>
      <head><title>Payment cancelled</title></head>
      <body style="font-family:Arial;padding:40px;text-align:center;background:#0b1020;color:white;">
        <h1>Payment cancelled</h1>
        <p>No credits were added.</p>
        <p>You can return to the app.</p>
      </body>
    </html>
    """)
@app.post("/register")
def register(email: str = Form(...), password: str = Form(...)):
    email = email.lower().strip()

    if "@" not in email or len(email) < 5:
        raise HTTPException(status_code=400, detail="Email invalide")

    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Mot de passe trop court (min 6 caractères)")

    existing = get_user_by_email(email)
    if existing:
        raise HTTPException(status_code=409, detail="Cet email existe déjà")

    user = create_user(email, password)
    token = create_access_token(user["id"], user["email"])

    return {
        "success": True,
        "message": "Compte créé",
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "email": user["email"],
            "credits": user["credits"],
            "created_at": user["created_at"],
        },
    }

@app.post("/login")
def login(email: str = Form(...), password: str = Form(...)):
    email = email.lower().strip()
    user = get_user_by_email(email)

    if not user or not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Identifiants invalides")

    token = create_access_token(user["id"], user["email"])

    return {
        "success": True,
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "email": user["email"],
            "credits": user["credits"],
            "created_at": user["created_at"],
        },
    }

@app.post("/create-checkout-session")
async def create_checkout_session(
    request: Request,
    user: sqlite3.Row = Depends(get_current_user),
):
    data = await request.json()
    pack = str(data.get("pack", "")).strip()

    selected_pack = CREDIT_PACKS.get(pack)
    if not selected_pack:
        raise HTTPException(status_code=400, detail="Invalid pack")

    price = int(selected_pack["price"])
    credits = int(selected_pack["credits"])
    label = str(selected_pack["label"]).strip()

    email = str(user["email"]).strip().lower()

    success_url = STRIPE_SUCCESS_URL
    if "{CHECKOUT_SESSION_ID}" not in success_url:
        separator = "&" if "?" in success_url else "?"
        success_url = f"{success_url}{separator}session_id={{CHECKOUT_SESSION_ID}}"

    session = stripe.checkout.Session.create(
        mode="payment",
        customer_email=email,
        payment_method_types=["card"],
        line_items=[
            {
                "price_data": {
                    "currency": "eur",
                    "product_data": {
                        "name": f"{credits} credits pack",
                    },
                    "unit_amount": price,
                },
                "quantity": 1,
            }
        ],
        success_url=success_url,
        cancel_url=STRIPE_CANCEL_URL,
        metadata={
            "user_email": email,
            "pack_key": pack,
            "credits_to_add": str(credits),
            "pack_name": label,
        },
    )

    return {
        "success": True,
        "url": session.url,
    }

# =========================================================
# ROUTES - PRIVATE
# =========================================================

@app.get("/me")
def me(user: sqlite3.Row = Depends(get_current_user)):
    return {
        "success": True,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "credits": user["credits"],
            "created_at": user["created_at"],
        },
    }

@app.get("/credits")
def credits(user: sqlite3.Row = Depends(get_current_user)):
    return {
        "success": True,
        "credits": user["credits"],
    }

@app.get("/credit-transactions")
def credit_transactions(user: sqlite3.Row = Depends(get_current_user)):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, transaction_type, amount, balance_after, stripe_payment_id, note, created_at
        FROM credit_transactions
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 50
        """,
        (user["id"],)
    )
    rows = cur.fetchall()
    conn.close()

    return {
        "success": True,
        "transactions": [dict(row) for row in rows]
    }

@app.post("/confirm-checkout-session")
def confirm_checkout_session(
    data: dict = Body(...),
    user: sqlite3.Row = Depends(get_current_user),
):
    session_id = normalize_checkout_session_id(data.get("session_id", ""))

    result = credit_paid_checkout_session(
        session_id,
        expected_email=str(user["email"]).strip().lower()
    )

    fresh_user = get_user_by_id(user["id"])
    credits_total = int(fresh_user["credits"]) if fresh_user else result["credits_total"]

    return {
        "success": True,
        "credits": credits_total,
        "credits_added": result["credits_added"],
        "credited_email": result["credited_email"],
        "session_id": session_id,
    }
@app.post("/stripe-webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(
            payload,
            sig_header,
            STRIPE_WEBHOOK_SECRET
        )
    except Exception as e:
        print("STRIPE SIGNATURE ERROR:")
        return {"status": "signature error"}

    event_type = event["type"]
    

    if event_type == "checkout.session.completed":
        try:
            raw_session = event["data"]["object"]

            session_id = normalize_checkout_session_id(
                stripe_obj_get(raw_session, "id", "")
            )

          

            if not session_id:
                print("WEBHOOK: session id manquant")
                return {"status": "ignored"}

            result = credit_paid_checkout_session(session_id)
            

        except Exception as e:
            import traceback
            print("WEBHOOK ERROR:")
            traceback.print_exc()
            return {"status": "error handled"}

    return {"status": "success"}
@app.post("/age")
async def age_face(
    request: Request,
    file: UploadFile = File(...),
    age: int = Form(...),
    user: sqlite3.Row = Depends(get_current_user),
):
    ip = client_ip(request)
    rate_key = f"age:{user['id']}:{ip}"
    check_rate_limit(rate_key)

    MAX_DAILY_REQUESTS = 50
    today = datetime.now(timezone.utc).date().isoformat()

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) as count FROM generation_logs
        WHERE user_id = ? AND DATE(created_at) = ?
    """, (user["id"], today))

    count = cur.fetchone()["count"]
    conn.close()

    if count >= MAX_DAILY_REQUESTS:
        raise HTTPException(status_code=429, detail="Limite journalière atteinte")

    if age < 1 or age > MAX_AGE:
        raise HTTPException(status_code=400, detail=f"L'âge doit être entre 1 et {MAX_AGE}")

    if int(user["credits"]) <= 0:
        return JSONResponse(
            status_code=402,
            content={
                "success": False,
                "error": "Crédits insuffisants",
                "code": "INSUFFICIENT_CREDITS"
            },
        )

    input_path = None
    output_filename = None

    try:
        os.environ["FAL_KEY"] = FAL_KEY

        content = await file.read()
        validate_uploaded_image(file, content)

        ext = Path(file.filename or "input.jpg").suffix.lower() or ".jpg"
        input_path = UPLOAD_DIR / f"input_{uuid4().hex[:8]}{ext}"
        input_path.write_bytes(content)

        uploaded_url = fal_client.upload_file(str(input_path))
        preserve_identity = age < 60

        result = fal_client.subscribe(
            MODEL_ID,
            arguments={
                "image_url": uploaded_url,
                "target_age": age,
                "preserve_identity": preserve_identity,
            },
        )

        images = result.get("images", [])
        if not images or not images[0].get("url"):
            raise RuntimeError("Réponse FAL invalide : aucune image retournée")

        image_url = images[0]["url"]

        response = requests.get(image_url, timeout=120)
        response.raise_for_status()
        img_bytes = response.content

        output_filename = f"aged_{age}_{uuid4().hex[:8]}.png"
        output_path = OUTPUT_DIR / output_filename
        output_path.write_bytes(img_bytes)

        new_credits = int(user["credits"]) - 1
        update_user_credits(user["id"], new_credits)

        add_credit_transaction(
            user_id=user["id"],
            transaction_type="usage",
            amount=-1,
            balance_after=new_credits,
            stripe_payment_id=None,
            note=f"Génération vieillissement âge cible {age}",
        )

        log_generation(
            user_id=user["id"],
            requested_age=age,
            credits_used=1,
            output_filename=output_filename,
            ip_address=ip,
            status="success",
            error_message=None,
        )

        return {
            "success": True,
            "image_url": f"{PUBLIC_BASE_URL}/outputs/{output_filename}",
            "file_path": str(output_path),
            "filename": output_filename,
            "age": age,
            "credits_remaining": new_credits,
        }

    except HTTPException:
        raise
    except Exception as e:
        log_generation(
            user_id=user["id"],
            requested_age=age,
            credits_used=0,
            output_filename=output_filename,
            ip_address=ip,
            status="error",
            error_message=str(e),
        )
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )
    finally:
        try:
            if input_path and input_path.exists():
                input_path.unlink(missing_ok=True)
        except Exception:
            pass

# =========================================================
# ADMIN / DEV TEMPORARY ROUTES
# =========================================================

@app.get("/debug/user-by-email")
def debug_user_by_email(email: str):
    user = get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")
    return {
        "id": user["id"],
        "email": user["email"],
        "credits": user["credits"],
        "created_at": user["created_at"],
    }

@app.post("/dev/add-credits")
def dev_add_credits(
    request: Request,
    email: str = Form(...),
    amount: int = Form(...),
):
    require_dev_admin(request)

    if amount <= 0:
        raise HTTPException(status_code=400, detail="Le montant doit être positif")

    user = get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")

    new_credits = int(user["credits"]) + amount
    update_user_credits(user["id"], new_credits)

    add_credit_transaction(
        user_id=user["id"],
        transaction_type="admin_add",
        amount=amount,
        balance_after=new_credits,
        stripe_payment_id=None,
        note="Ajout manuel développement",
    )

    return {
        "success": True,
        "message": "Crédits ajoutés",
        "email": user["email"],
        "credits_added": amount,
        "credits_total": new_credits,
    }
