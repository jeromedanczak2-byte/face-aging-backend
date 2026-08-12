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
import asyncio
import json
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
        "price": 3900,
        "credits": 30,
        "label": "Starter",
    },
    "100": {
        "price": 11500,
        "credits": 100,
        "label": "Best Seller",
    },
    "300": {
        "price": 29900,
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
GA4_MEASUREMENT_ID = "G-FDGNC88XDV"
GA4_API_SECRET = os.getenv("GA4_API_SECRET", "").strip()
WINDOWS_ANALYTICS_EVENTS = {
    "windows_app_open",
    "login",
    "begin_checkout",
    "generation_start",
    "generation_success",
}
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

DEMO_FREE_LIMIT = int(os.getenv("DEMO_FREE_LIMIT", "1"))
DEMO_TARGET_AGE = int(os.getenv("DEMO_TARGET_AGE", "90"))
DEMO_OUTPUT_MAX_WIDTH = int(os.getenv("DEMO_OUTPUT_MAX_WIDTH", "360"))
DEMO_OUTPUT_QUALITY = int(os.getenv("DEMO_OUTPUT_QUALITY", "40"))
DEMO_WATERMARK_TEXT = os.getenv("DEMO_WATERMARK_TEXT", "DEMO PREVIEW").strip()
MAX_WEB_BATCH_AGES = int(os.getenv("MAX_WEB_BATCH_AGES", "5"))

CORS_ORIGINS_RAW = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:1420,http://127.0.0.1:1420,http://tauri.localhost,https://tauri.localhost,https://faceagingstudio.com,https://www.faceagingstudio.com"
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

    cur.execute("""
        CREATE TABLE IF NOT EXISTS demo_generations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ip_address TEXT,
            client_token_hash TEXT,
            output_filename TEXT,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL,
            error_message TEXT
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_demo_generations_ip_status
        ON demo_generations(ip_address, status)
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_demo_generations_client_status
        ON demo_generations(client_token_hash, status)
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


def reserve_user_credits(user_id: int, credits_needed: int, note: str) -> dict:
    if credits_needed <= 0:
        raise ValueError("credits_needed must be positive")

    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        cur.execute("SELECT credits FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()

        if not row:
            conn.rollback()
            raise HTTPException(status_code=401, detail="Utilisateur introuvable")

        current_credits = int(row["credits"])
        if current_credits < credits_needed:
            conn.rollback()
            raise HTTPException(
                status_code=402,
                detail={
                    "success": False,
                    "error": "Crédits insuffisants",
                    "code": "INSUFFICIENT_CREDITS",
                    "credits_available": current_credits,
                    "credits_needed": credits_needed,
                },
            )

        new_credits = current_credits - credits_needed
        cur.execute("UPDATE users SET credits = ? WHERE id = ?", (new_credits, user_id))
        cur.execute(
            """
            INSERT INTO credit_transactions
            (user_id, transaction_type, amount, balance_after, stripe_payment_id, note, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                "usage",
                -credits_needed,
                new_credits,
                None,
                note,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return {
            "credits_before": current_credits,
            "credits_after": new_credits,
            "credits_reserved": credits_needed,
        }
    except HTTPException:
        raise
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def refund_user_credits(user_id: int, credits_to_refund: int, note: str) -> int:
    if credits_to_refund <= 0:
        fresh_user = get_user_by_id(user_id)
        return int(fresh_user["credits"]) if fresh_user else 0

    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        cur.execute("SELECT credits FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()

        if not row:
            conn.rollback()
            return 0

        current_credits = int(row["credits"])
        new_credits = current_credits + credits_to_refund
        cur.execute("UPDATE users SET credits = ? WHERE id = ?", (new_credits, user_id))
        cur.execute(
            """
            INSERT INTO credit_transactions
            (user_id, transaction_type, amount, balance_after, stripe_payment_id, note, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                "refund",
                credits_to_refund,
                new_credits,
                None,
                note,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return new_credits
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def parse_web_batch_ages(raw_ages: str) -> list[int]:
    raw_ages = (raw_ages or "").strip()
    if not raw_ages:
        raise HTTPException(status_code=400, detail="Aucun âge sélectionné")

    try:
        if raw_ages.startswith("["):
            parsed = json.loads(raw_ages)
        else:
            parsed = [value.strip() for value in raw_ages.split(",")]
    except Exception:
        raise HTTPException(status_code=400, detail="Format des âges invalide")

    ages: list[int] = []
    for value in parsed:
        try:
            age = int(value)
        except Exception:
            raise HTTPException(status_code=400, detail="Format des âges invalide")

        if age < 1 or age > MAX_AGE:
            raise HTTPException(status_code=400, detail=f"L'âge doit être entre 1 et {MAX_AGE}")

        if age not in ages:
            ages.append(age)

    if not ages:
        raise HTTPException(status_code=400, detail="Aucun âge sélectionné")

    if len(ages) > MAX_WEB_BATCH_AGES:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_WEB_BATCH_AGES} âges par génération web mobile",
        )

    return ages

def generate_paid_web_age_from_uploaded_url(uploaded_url: str, age: int, ip: Optional[str]) -> dict:
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

    output_filename = f"web_aged_{age}_{uuid4().hex[:8]}.png"
    output_path = OUTPUT_DIR / output_filename
    output_path.write_bytes(response.content)

    return {
        "success": True,
        "age": age,
        "image_url": f"{PUBLIC_BASE_URL}/outputs/{output_filename}",
        "file_path": str(output_path),
        "filename": output_filename,
    }


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

def normalize_demo_client_token(value: Optional[str]) -> str:
    cleaned = str(value or "").strip()
    return cleaned[:128]

def hash_demo_client_token(value: str) -> Optional[str]:
    if not value:
        return None
    secret = JWT_SECRET or "face-aging-demo"
    return hashlib.sha256(f"{secret}:{value}".encode("utf-8")).hexdigest()

def get_demo_success_count(ip_address: str, client_token_hash: Optional[str]) -> int:
    conn = get_db()
    cur = conn.cursor()

    if client_token_hash:
        cur.execute(
            """
            SELECT COUNT(*) as count FROM demo_generations
            WHERE status = 'success'
            AND (ip_address = ? OR client_token_hash = ?)
            """,
            (ip_address, client_token_hash),
        )
    else:
        cur.execute(
            """
            SELECT COUNT(*) as count FROM demo_generations
            WHERE status = 'success'
            AND ip_address = ?
            """,
            (ip_address,),
        )

    count = int(cur.fetchone()["count"])
    conn.close()
    return count

def check_demo_limit(ip_address: str, client_token_hash: Optional[str]):
    count = get_demo_success_count(ip_address, client_token_hash)
    if count >= DEMO_FREE_LIMIT:
        return JSONResponse(
            status_code=429,
            content={
                "success": False,
                "error": "Demo limit reached",
                "code": "DEMO_LIMIT_REACHED",
            },
        )
    return None

def log_demo_generation(
    ip_address: str,
    client_token_hash: Optional[str],
    output_filename: Optional[str],
    status: str,
    error_message: Optional[str] = None,
):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO demo_generations
        (ip_address, client_token_hash, output_filename, created_at, status, error_message)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            ip_address,
            client_token_hash,
            output_filename,
            datetime.now(timezone.utc).isoformat(),
            status,
            error_message,
        ),
    )
    conn.commit()
    conn.close()

def make_demo_watermarked_image(img_bytes: bytes) -> bytes:
    try:
        from io import BytesIO
        from PIL import Image, ImageDraw, ImageFont
    except Exception as e:
        raise RuntimeError("Pillow is required for demo watermark") from e

    def load_demo_font(size: int):
        font_candidates = [
            "DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "arial.ttf",
            "Arial.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
        for font_path in font_candidates:
            try:
                return ImageFont.truetype(font_path, size=size)
            except Exception:
                continue
        return ImageFont.load_default()

    def text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
        try:
            box = draw.textbbox((0, 0), text, font=font, stroke_width=2)
            return box[2] - box[0], box[3] - box[1]
        except Exception:
            return len(text) * 9, 18

    def fitted_font(draw: ImageDraw.ImageDraw, text: str, max_width: int, start_size: int, min_size: int):
        size = start_size
        while size > min_size:
            font = load_demo_font(size)
            width, _ = text_size(draw, text, font)
            if width <= max_width:
                return font
            size -= 2
        return load_demo_font(min_size)

    with Image.open(BytesIO(img_bytes)) as img:
        img = img.convert("RGB")
        try:
            resample_filter = Image.Resampling.LANCZOS
            rotate_filter = Image.Resampling.BICUBIC
        except AttributeError:
            resample_filter = Image.LANCZOS
            rotate_filter = Image.BICUBIC

        img.thumbnail((DEMO_OUTPUT_MAX_WIDTH, DEMO_OUTPUT_MAX_WIDTH), resample_filter)

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        watermark_text = DEMO_WATERMARK_TEXT or "DEMO PREVIEW"
        site_text = "FACEAGINGSTUDIO.COM"
        max_text_width = int(img.width * 0.72)

        primary_font = fitted_font(
            draw,
            watermark_text,
            max_text_width,
            start_size=max(20, min(img.width // 9, 34)),
            min_size=14,
        )
        secondary_font = fitted_font(
            draw,
            site_text,
            max_text_width,
            start_size=max(11, min(img.width // 18, 18)),
            min_size=9,
        )

        primary_w, primary_h = text_size(draw, watermark_text, primary_font)
        site_w, site_h = text_size(draw, site_text, secondary_font)

        box_padding_x = max(14, img.width // 24)
        box_padding_y = max(9, img.height // 44)
        box_w = img.width - (max(10, img.width // 28) * 2)
        box_h = primary_h + site_h + (box_padding_y * 2) + 8
        box_x = (img.width - box_w) // 2
        box_y = img.height - box_h - max(10, img.height // 32)
        box_radius = max(12, min(img.width, img.height) // 22)

        draw.rounded_rectangle(
            (box_x, box_y, box_x + box_w, box_y + box_h),
            radius=box_radius,
            fill=(0, 0, 0, 118),
            outline=(255, 255, 255, 58),
            width=1,
        )

        primary_x = box_x + (box_w - primary_w) // 2
        primary_y = box_y + box_padding_y - 2
        site_x = box_x + (box_w - site_w) // 2
        site_y = primary_y + primary_h + 8

        draw.text(
            (primary_x, primary_y),
            watermark_text,
            fill=(255, 255, 255, 235),
            font=primary_font,
            stroke_width=1,
            stroke_fill=(0, 0, 0, 130),
        )
        draw.text(
            (site_x, site_y),
            site_text,
            fill=(255, 255, 255, 205),
            font=secondary_font,
            stroke_width=1,
            stroke_fill=(0, 0, 0, 110),
        )

        diagonal_text = " FACEAGINGSTUDIO.COM DEMO "
        diagonal_font = fitted_font(
            draw,
            diagonal_text,
            int(img.width * 0.95),
            start_size=max(11, min(img.width // 20, 18)),
            min_size=9,
        )
        diagonal_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        diagonal_draw = ImageDraw.Draw(diagonal_layer)
        diagonal_w, diagonal_h = text_size(diagonal_draw, diagonal_text, diagonal_font)
        step_y = max(72, diagonal_h * 3)
        step_x = max(80, diagonal_w // 2)
        for y in range(-img.height, img.height * 2, step_y):
            for x in range(-img.width, img.width * 2, step_x):
                diagonal_draw.text(
                    (x, y),
                    diagonal_text,
                    fill=(255, 255, 255, 34),
                    font=diagonal_font,
                    stroke_width=1,
                    stroke_fill=(0, 0, 0, 22),
                )
        diagonal_layer = diagonal_layer.rotate(-24, resample=rotate_filter, expand=False, center=(img.width // 2, img.height // 2))
        overlay = Image.alpha_composite(overlay, diagonal_layer)

        border_width = max(2, img.width // 120)
        border_draw = ImageDraw.Draw(overlay)
        for i in range(border_width):
            border_draw.rectangle((i, i, img.width - 1 - i, img.height - 1 - i), outline=(255, 255, 255, 22))

        watermarked = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        output = BytesIO()
        watermarked.save(output, format="JPEG", quality=DEMO_OUTPUT_QUALITY, optimize=True, subsampling=2)
        return output.getvalue()

def validate_uploaded_image(file: UploadFile, content: bytes):
    ext = Path(file.filename or "input.jpg").suffix.lower()
    content_type = (file.content_type or "").lower()

    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Extension non autorisée. Formats acceptés: jpg, jpeg, png, webp"
        )

    if content_type and content_type not in ALLOWED_IMAGE_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Type MIME non autorisé"
        )

    if not content:
        raise HTTPException(status_code=400, detail="Fichier vide")

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Fichier trop volumineux. Maximum {MAX_UPLOAD_MB} MB"
        )

def stripe_obj_get(obj: Any, key: str, default=None):
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(key, default)

    try:
        value = obj[key]
        if value is None:
            return default
        return value
    except Exception:
        pass

    try:
        value = getattr(obj, key)
        if value is None:
            return default
        return value
    except Exception:
        return default

def normalize_checkout_session_id(raw_value: Any) -> str:
    raw = str(raw_value or "").strip()

    if not raw:
        return ""

    for separator in ["&", "?", ";", " ", "\n", "\r", "\t"]:
        if separator in raw:
            raw = raw.split(separator)[0].strip()

    if "cs_" in raw and not raw.startswith("cs_"):
        raw = raw[raw.find("cs_"):].strip()
        for separator in ["&", "?", ";", " ", "\n", "\r", "\t"]:
            if separator in raw:
                raw = raw.split(separator)[0].strip()

    return raw

def credit_paid_checkout_session(session_id: str, expected_email: Optional[str] = None) -> dict:
    session_id = normalize_checkout_session_id(session_id)

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id manquant")

    if not session_id.startswith("cs_"):
        raise HTTPException(status_code=400, detail="session_id Stripe invalide")

    existing_tx = get_credit_transaction_by_stripe_payment_id(session_id)
    if existing_tx:
        user = get_user_by_id(existing_tx["user_id"])
        return {
            "success": True,
            "message": "Session déjà traitée",
            "credited_email": user["email"] if user else "",
            "credits_added": 0,
            "credits_total": int(user["credits"]) if user else 0,
        }

    try:
        session = stripe.checkout.Session.retrieve(session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Impossible de lire la session Stripe: {str(e)}")

    mode = stripe_obj_get(session, "mode", "")
    if mode != "payment":
        raise HTTPException(status_code=400, detail="Mode Stripe invalide")

    amount_total = int(stripe_obj_get(session, "amount_total", 0))
    payment_status = str(stripe_obj_get(session, "payment_status", "")).strip().lower()
    if payment_status != "paid":
        raise HTTPException(status_code=400, detail="Paiement non confirmé")

    metadata = stripe_obj_get(session, "metadata", {}) or {}
    email = str(stripe_obj_get(metadata, "user_email", "")).strip().lower()

    if not email:
        customer_email = stripe_obj_get(session, "customer_email", "")
        email = str(customer_email).strip().lower()

    if not email:
        customer_details = stripe_obj_get(session, "customer_details", {}) or {}
        email = str(stripe_obj_get(customer_details, "email", "")).strip().lower()

    if not email:
        raise HTTPException(status_code=400, detail="Email introuvable dans la session Stripe")

    if expected_email and email != expected_email.lower().strip():
        raise HTTPException(status_code=403, detail="Cette session Stripe n'appartient pas à cet utilisateur")

    pack_key = str(stripe_obj_get(metadata, "pack_key", "")).strip()
    selected_pack = CREDIT_PACKS.get(pack_key)
    if not selected_pack:
        raise HTTPException(status_code=400, detail="Pack Stripe invalide")

    expected_amount = int(selected_pack["price"])
    if amount_total != expected_amount:
        raise HTTPException(status_code=400, detail="Montant Stripe invalide")

    credits_to_add = int(selected_pack["credits"])
    pack_name = str(selected_pack["label"]).strip()

    if credits_to_add <= 0:
        raise HTTPException(status_code=400, detail="credits_to_add invalide")

    user = get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=404, detail=f"Utilisateur introuvable pour {email}")

    conn = get_db()
    try:
        cur = conn.cursor()

        cur.execute(
            "SELECT * FROM credit_transactions WHERE stripe_payment_id = ? LIMIT 1",
            (session_id,)
        )
        existing_tx_in_conn = cur.fetchone()
        if existing_tx_in_conn:
            cur.execute("SELECT * FROM users WHERE id = ?", (existing_tx_in_conn["user_id"],))
            existing_user = cur.fetchone()
            return {
                "success": True,
                "message": "Session déjà traitée",
                "credited_email": existing_user["email"] if existing_user else "",
                "credits_added": 0,
                "credits_total": int(existing_user["credits"]) if existing_user else 0,
            }

        cur.execute("SELECT credits FROM users WHERE id = ?", (user["id"],))
        fresh_user = cur.fetchone()
        if not fresh_user:
            raise HTTPException(status_code=404, detail="Utilisateur introuvable")

        current_credits = int(fresh_user["credits"])
        new_credits = current_credits + credits_to_add

        cur.execute(
            "UPDATE users SET credits = ? WHERE id = ?",
            (new_credits, user["id"])
        )

        cur.execute(
            """
            INSERT INTO credit_transactions
            (user_id, transaction_type, amount, balance_after, stripe_payment_id, note, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user["id"],
                "purchase",
                credits_to_add,
                new_credits,
                session_id,
                f"Paiement Stripe - {pack_name}",
                datetime.now(timezone.utc).isoformat(),
            )
        )

        conn.commit()
    except sqlite3.IntegrityError:
        conn.rollback()
        user_after = get_user_by_id(user["id"])
        return {
            "success": True,
            "message": "Session déjà traitée",
            "credited_email": user_after["email"] if user_after else email,
            "credits_added": 0,
            "credits_total": int(user_after["credits"]) if user_after else int(user["credits"]),
        }
    finally:
        conn.close()

    return {
        "success": True,
        "credited_email": email,
        "credits_added": credits_to_add,
        "credits_total": new_credits,
    }

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

    if GA4_API_SECRET:
        try:
            requests.post(
                "https://www.google-analytics.com/mp/collect",
                params={
                    "measurement_id": GA4_MEASUREMENT_ID,
                    "api_secret": GA4_API_SECRET,
                },
                json={
                    "client_id": f"fas.{user['id']}",
                    "user_id": str(user["id"]),
                    "events": [
                        {
                            "name": "sign_up",
                            "params": {
                                "method": "email",
                                "session_id": int(time.time()),
                                "engagement_time_msec": 1,
                            },
                        }
                    ],
                },
                timeout=3,
            )
        except Exception as e:
            print("GA4 SIGN_UP ERROR:", e)

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

@app.post("/analytics/windows")
async def analytics_windows(request: Request):
    data = await request.json()
    event_name = str(data.get("event", "")).strip()
    client_id = str(data.get("client_id", "")).strip()[:128]

    if event_name not in WINDOWS_ANALYTICS_EVENTS:
        raise HTTPException(status_code=400, detail="Invalid analytics event")

    if not client_id:
        raise HTTPException(status_code=400, detail="Missing analytics client_id")

    user_id = None
    if event_name != "windows_app_open":
        authorization = request.headers.get("authorization", "").strip()
        if not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Missing analytics authorization")

        payload = decode_access_token(authorization.split(" ", 1)[1].strip())
        user_id = int(payload["sub"])
        if not get_user_by_id(user_id):
            raise HTTPException(status_code=401, detail="Utilisateur introuvable")

    if not GA4_API_SECRET:
        return {"success": True, "sent": False}

    ga4_payload = {
        "client_id": client_id,
        "events": [
            {
                "name": event_name,
                "params": {
                    "surface": "windows_app",
                    "session_id": int(time.time()),
                    "engagement_time_msec": 1,
                },
            }
        ],
    }

    if user_id is not None:
        ga4_payload["user_id"] = str(user_id)

    try:
        response = requests.post(
            "https://www.google-analytics.com/mp/collect",
            params={
                "measurement_id": GA4_MEASUREMENT_ID,
                "api_secret": GA4_API_SECRET,
            },
            json=ga4_payload,
            timeout=3,
        )
        response.raise_for_status()
        return {"success": True, "sent": True}
    except Exception as e:
        print("GA4 WINDOWS ANALYTICS ERROR:", e)
        return {"success": True, "sent": False}

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
        payment_method_types=["card", "mobilepay"],
        line_items=[
            {
                "price_data": {
                    "currency": "dkk",
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


@app.post("/web/age-batch")
async def web_age_batch(
    request: Request,
    file: UploadFile = File(...),
    ages: str = Form(...),
    user: sqlite3.Row = Depends(get_current_user),
):
    ip = client_ip(request)
    user_id = int(user["id"])
    requested_ages = parse_web_batch_ages(ages)
    credits_needed = len(requested_ages)

    rate_key = f"web-age-batch:{user_id}:{ip}"
    check_rate_limit(rate_key)

    input_path = None
    reserved = False
    reserved_count = 0

    try:
        content = await file.read()
        validate_uploaded_image(file, content)

        reserve_user_credits(
            user_id=user_id,
            credits_needed=credits_needed,
            note=f"Réservation génération web mobile âges {', '.join(map(str, requested_ages))}",
        )
        reserved = True
        reserved_count = credits_needed

        os.environ["FAL_KEY"] = FAL_KEY

        ext = Path(file.filename or "input.jpg").suffix.lower() or ".jpg"
        input_path = UPLOAD_DIR / f"web_batch_input_{uuid4().hex[:8]}{ext}"
        input_path.write_bytes(content)

        uploaded_url = fal_client.upload_file(str(input_path))

        tasks = [
            asyncio.to_thread(generate_paid_web_age_from_uploaded_url, uploaded_url, age, ip)
            for age in requested_ages
        ]

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        success_count = 0
        failed_count = 0

        for age, raw_result in zip(requested_ages, raw_results):
            if isinstance(raw_result, Exception):
                failed_count += 1
                error_message = str(raw_result)
                log_generation(
                    user_id=user_id,
                    requested_age=age,
                    credits_used=0,
                    output_filename=None,
                    ip_address=ip,
                    status="error",
                    error_message=error_message,
                )
                results.append({
                    "success": False,
                    "age": age,
                    "error": error_message,
                })
                continue

            success_count += 1
            log_generation(
                user_id=user_id,
                requested_age=age,
                credits_used=1,
                output_filename=raw_result.get("filename"),
                ip_address=ip,
                status="success",
                error_message=None,
            )
            results.append(raw_result)

        if failed_count > 0:
            credits_remaining = refund_user_credits(
                user_id=user_id,
                credits_to_refund=failed_count,
                note=f"Remboursement génération web mobile échouée : {failed_count} crédit(s)",
            )
        else:
            fresh_user = get_user_by_id(user_id)
            credits_remaining = int(fresh_user["credits"]) if fresh_user else 0

        return {
            "success": success_count > 0,
            "mode": "web_batch",
            "ages": requested_ages,
            "credits_used": success_count,
            "credits_refunded": failed_count,
            "credits_remaining": credits_remaining,
            "results": results,
        }

    except HTTPException:
        raise
    except Exception as e:
        if reserved and reserved_count > 0:
            try:
                refund_user_credits(
                    user_id=user_id,
                    credits_to_refund=reserved_count,
                    note="Remboursement génération web mobile après erreur globale",
                )
            except Exception as refund_error:
                print("WEB BATCH REFUND ERROR:", refund_error)

        for age in requested_ages:
            log_generation(
                user_id=user_id,
                requested_age=age,
                credits_used=0,
                output_filename=None,
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
# ROUTES - WEB DEMO / ONLINE APP
# =========================================================

@app.post("/web/demo-age")
async def web_demo_age(
    request: Request,
    file: UploadFile = File(...),
    age: int = Form(...),
    demo_client: str = Form(""),
):
    ip = client_ip(request)
    clean_demo_client = normalize_demo_client_token(demo_client)
    client_token_hash = hash_demo_client_token(clean_demo_client)

    rate_key = f"web-demo-age:{ip}:{client_token_hash or 'no-client'}"
    check_rate_limit(rate_key)

    limit_response = check_demo_limit(ip, client_token_hash)
    if limit_response:
        return limit_response

    input_path = None
    output_filename = None

    if age < 1 or age > MAX_AGE:
        raise HTTPException(status_code=400, detail=f"L'âge doit être entre 1 et {MAX_AGE}")

    try:
        os.environ["FAL_KEY"] = FAL_KEY

        content = await file.read()
        validate_uploaded_image(file, content)

        ext = Path(file.filename or "input.jpg").suffix.lower() or ".jpg"
        input_path = UPLOAD_DIR / f"demo_input_{uuid4().hex[:8]}{ext}"
        input_path.write_bytes(content)

        uploaded_url = fal_client.upload_file(str(input_path))

        result = fal_client.subscribe(
            MODEL_ID,
            arguments={
                "image_url": uploaded_url,
                "target_age": age,
                "preserve_identity": False,
            },
        )

        images = result.get("images", [])
        if not images or not images[0].get("url"):
            raise RuntimeError("Réponse FAL invalide : aucune image retournée")

        image_url = images[0]["url"]

        response = requests.get(image_url, timeout=120)
        response.raise_for_status()
        demo_img_bytes = make_demo_watermarked_image(response.content)

        output_filename = f"demo_aged_{age}_{uuid4().hex[:8]}.jpg"
        output_path = OUTPUT_DIR / output_filename
        output_path.write_bytes(demo_img_bytes)

        log_demo_generation(
            ip_address=ip,
            client_token_hash=client_token_hash,
            output_filename=output_filename,
            status="success",
            error_message=None,
        )

        return {
            "success": True,
            "demo": True,
            "watermark": True,
            "low_resolution": True,
            "image_url": f"{PUBLIC_BASE_URL}/outputs/{output_filename}",
            "filename": output_filename,
            "age": age,
        }

    except HTTPException:
        raise
    except Exception as e:
        log_demo_generation(
            ip_address=ip,
            client_token_hash=client_token_hash,
            output_filename=output_filename,
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
