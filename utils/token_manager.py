# utils/token_manager.py
"""Keeps the Twitch OAuth access token fresh.

A background thread (see chatbot.start_token_management) calls
manage_token_lifecycle, which periodically refreshes the access token using the
stored refresh token before it expires.  Tokens live in config.TOKEN_FILE; the
client id/secret come from the environment via config.
"""

import json
import logging
import time

import requests

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TOKEN_URL = "https://id.twitch.tv/oauth2/token"


def read_token_data() -> dict:
    with open(config.TOKEN_FILE, "r") as f:
        return json.load(f)


def update_token_data(new_tokens: dict) -> None:
    data = {
        "access_token": "oauth:" + new_tokens["access_token"],
        "refresh_token": new_tokens["refresh_token"],
        # keep as float for easy comparisons; refresh 5 minutes before expiry
        "expiry_time": float(time.time() + float(new_tokens.get("expires_in", 0)) - 300.0),
    }
    with open(config.TOKEN_FILE, "w") as f:
        json.dump(data, f)
    logging.info("Token data updated.")


def refresh_access_token() -> bool:
    """Refresh the Twitch access token using the stored refresh token."""
    token_data = read_token_data()
    client_id = config.TWITCH_CLIENT_ID
    client_secret = config.TWITCH_CLIENT_SECRET

    if not client_id or not client_secret:
        logging.error("Cannot refresh: TWITCH_CLIENT_ID or TWITCH_CLIENT_SECRET is missing.")
        return False

    payload = {
        "grant_type": "refresh_token",
        "refresh_token": token_data["refresh_token"],
        "client_id": client_id,
        "client_secret": client_secret,
    }

    try:
        resp = requests.post(TOKEN_URL, data=payload, timeout=15)
    except Exception as e:
        logging.error(f"Token refresh HTTP error: {e}")
        return False

    if resp.status_code == 200:
        update_token_data(resp.json())
        logging.info("Access token refreshed successfully.")
        return True

    try:
        logging.error(f"Failed to refresh token ({resp.status_code}): {resp.text}")
    except Exception:
        logging.error(f"Failed to refresh token ({resp.status_code}).")
    return False


def check_and_refresh_token() -> None:
    token_data = read_token_data()
    try:
        expiry = float(token_data["expiry_time"])
    except (KeyError, ValueError, TypeError):
        logging.warning("Expiry missing/invalid; forcing refresh.")
        refresh_access_token()
        return

    if time.time() > expiry:
        logging.info("Token expired — refreshing...")
        refresh_access_token()
    else:
        logging.info("Token still valid.")


def manage_token_lifecycle() -> None:
    logging.info("Starting token management loop.")
    while True:
        try:
            check_and_refresh_token()
        except Exception as e:
            logging.error(f"Error managing token lifecycle: {e}")
        time.sleep(300)  # 5 minutes


def get_current_helix_token() -> str:
    """Return the latest access token for the Helix API (no 'oauth:' prefix)."""
    token_data = read_token_data()
    token = token_data["access_token"]
    return token[6:] if token.startswith("oauth:") else token
