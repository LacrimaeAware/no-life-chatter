"""One-time helper to fetch the bot's initial Twitch OAuth tokens.

This replaces the old hosted OAuth redirect with a purely local flow, so you
don't need a public web server just to log in.

How it works:
  1. Opens the Twitch authorize page in your browser.
  2. Spins up a tiny local web server to catch the redirect with the auth code.
  3. Exchanges the code for an access + refresh token.
  4. Writes them to config.TOKEN_FILE in the format the bot expects.

Afterwards, utils/token_manager.py keeps the access token refreshed on its own.

Prerequisites:
  * TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET set in .env
  * Your Twitch app (https://dev.twitch.tv/console/apps) must list the redirect
    URL below under "OAuth Redirect URLs" (default: http://localhost:3000).

Usage:
    python scripts/get_initial_token.py
"""

import json
import os
import sys
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

REDIRECT_URL = os.getenv("OAUTH_REDIRECT_URL", "http://localhost:3000")
AUTHORIZE_URL = "https://id.twitch.tv/oauth2/authorize"
TOKEN_URL = "https://id.twitch.tv/oauth2/token"

# Scopes: read + send chat, and send whispers (used by practice mode).
SCOPES = ["chat:read", "chat:edit", "user:manage:whispers"]

_auth_code = {"code": None, "error": None}


class _CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        _auth_code["code"] = params.get("code", [None])[0]
        _auth_code["error"] = params.get("error_description", params.get("error", [None]))[0]

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        if _auth_code["code"]:
            body = "<h1>Authorization received.</h1><p>You can close this tab.</p>"
        else:
            body = f"<h1>Authorization failed.</h1><p>{_auth_code['error']}</p>"
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, *args):  # silence the default request logging
        pass


def _wait_for_code(host: str, port: int):
    server = HTTPServer((host, port), _CallbackHandler)
    print(f"Listening on {REDIRECT_URL} for the Twitch redirect...")
    while _auth_code["code"] is None and _auth_code["error"] is None:
        server.handle_request()
    server.server_close()


def main():
    if not config.TWITCH_CLIENT_ID or not config.TWITCH_CLIENT_SECRET:
        sys.exit("TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET missing. Set them in .env first.")

    parsed = urllib.parse.urlparse(REDIRECT_URL)
    host = parsed.hostname or "localhost"
    port = parsed.port or 80

    authorize_params = {
        "response_type": "code",
        "client_id": config.TWITCH_CLIENT_ID,
        "redirect_uri": REDIRECT_URL,
        "scope": " ".join(SCOPES),
    }
    url = f"{AUTHORIZE_URL}?{urllib.parse.urlencode(authorize_params)}"

    print("Opening the Twitch authorization page in your browser...")
    print(f"If it doesn't open, visit:\n  {url}\n")
    webbrowser.open(url)

    _wait_for_code(host, port)

    if _auth_code["error"]:
        sys.exit(f"Authorization failed: {_auth_code['error']}")

    print("Exchanging authorization code for tokens...")
    resp = requests.post(
        TOKEN_URL,
        data={
            "client_id": config.TWITCH_CLIENT_ID,
            "client_secret": config.TWITCH_CLIENT_SECRET,
            "code": _auth_code["code"],
            "grant_type": "authorization_code",
            "redirect_uri": REDIRECT_URL,
        },
        timeout=15,
    )
    if resp.status_code != 200:
        sys.exit(f"Token exchange failed ({resp.status_code}): {resp.text}")

    tokens = resp.json()
    data = {
        "access_token": "oauth:" + tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
        "expiry_time": float(time.time() + float(tokens.get("expires_in", 0)) - 300.0),
    }

    os.makedirs(os.path.dirname(config.TOKEN_FILE), exist_ok=True)
    with open(config.TOKEN_FILE, "w") as f:
        json.dump(data, f)

    print(f"\nSaved tokens to {config.TOKEN_FILE}")
    print("You can now start the bot with: python chatbot.py")


if __name__ == "__main__":
    main()
