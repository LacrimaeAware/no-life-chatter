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


def check_and_refresh_token() -> bool:
    """Refresh the token if it is at/near expiry. Returns True iff a refresh
    actually happened (so callers can push the new token into a live session)."""
    token_data = read_token_data()
    try:
        expiry = float(token_data["expiry_time"])
    except (KeyError, ValueError, TypeError):
        logging.warning("Expiry missing/invalid; forcing refresh.")
        return refresh_access_token()

    if time.time() > expiry:
        logging.info("Token expired — refreshing...")
        return refresh_access_token()
    logging.info("Token still valid.")
    return False


def manage_token_lifecycle(on_refresh=None) -> None:
    """Background loop: refresh the file token before it expires. When a refresh
    happens, call on_refresh() so the live bot can adopt the new token (twitchio
    otherwise keeps re-sending the token captured at startup — see
    chatbot.Bot.apply_current_token)."""
    logging.info("Starting token management loop.")
    while True:
        try:
            if check_and_refresh_token() and on_refresh:
                try:
                    on_refresh()
                except Exception as e:
                    logging.error(f"Token on_refresh callback failed: {e}")
        except Exception as e:
            logging.error(f"Error managing token lifecycle: {e}")
        time.sleep(300)  # 5 minutes


def get_current_helix_token() -> str:
    """Return the latest access token for the Helix API (no 'oauth:' prefix)."""
    token_data = read_token_data()
    token = token_data["access_token"]
    return token[6:] if token.startswith("oauth:") else token
