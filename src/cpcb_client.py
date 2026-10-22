"""Reusable CPCB CCR (airquality.cpcb.gov.in) API client.

Reverse-engineered transport (verified against captured live traffic on 2026-10-22):
- The SPA posts plain-text bodies to
  https://airquality.cpcb.gov.in/insight_proxy_api_v2_0/service.php
  with header `index: iv`.
- The body is CryptoJS-style AES-256-CBC with an OpenSSL "Salted__" frame:
  base64(b"Salted__" + 8-byte salt + ciphertext), where key/IV are derived by
  EVP_BytesToKey(passphrase, salt, md5, 1 iteration; key 32 bytes, IV 16 bytes)
  using passphrase "P7#Vw!3Xz@8Lm$2QkT9".
- The decrypted body is a JSON envelope:
    {methodType, url, header:{Authorization, Content-Type}, payload?, enableEncryption}
  where `url` is the real data endpoint on airquality.cpcb.gov.in and
  Authorization is a session token GUID.
- Server response bodies are AES-encrypted the same way (some return cleartext JSON).

Crypto facts recovered from /ccr/ chunk-WKM2WSKQ.js (deobfuscated):
  Le = 'iv' (header value), Qf = '7m67CrnRRldWUD2j' (unused at runtime),
  Fr.SK = 'Kz3TUIj7YQ' (OAuth bootstrap), runtime AES secretKey =
  'P7#Vw!3Xz@8Lm$2QkT9' (verified by decrypting live captures).

Session tokens expire after a few hundred data calls (guid 401s).  The SPA
re-mints them itself via getTokenFromDefaultCredentials(): an oauth envelope
{"methodType":"POST","url":"https://airquality.cpcb.gov.in/api/security/oauth/"
"token?grant_type=password&username=...&client_id=<Fr.Client3>&client_secret="
"<Fr.Sec3>","payload":"{}","enableEncryption":true} encrypted with passphrase
'iv' (NOT the data key), POSTed to service.php; the response is decrypted with
'iv' and contains access_token/refresh_token.  bootstrap_token() below does
exactly that, and call() re-bootstraps automatically once on 401/403.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os as _os
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

import requests

SERVICE_URL = "https://airquality.cpcb.gov.in/insight_proxy_api_v2_0/service.php"
BASE = "https://airquality.cpcb.gov.in/"
QUERY = BASE + "projectworkflow/client/custom/query/"
SK = "P7#Vw!3Xz@8Lm$2QkT9"
HEADER_NAME = "index"
HEADER_VALUE = "iv"

# OAuth bootstrap (SPA config Fr; Client1/2/3 and Sec1/2/3 are identical).
OAUTH_TOKEN_PATH = (
    "api/security/oauth/token?grant_type=password&username=ayush.gaur@mapmyindia.com"
    "&password=ab3edead07f87462ae80892589f91907&client_id="
)
OAUTH_CLIENT_ID = (
    "96dHZVzsAuv39kJWtHT6562eYTLXHPXWAFAjck3PKa7mZf5elLhA12C0_-QkRmtQGm3BurNxTcAhKF3878OZIAK311Rdi0Kz"
)
OAUTH_CLIENT_SECRET = (
    "lrFxI-iSEg9nf2wrYQExmAE16fBV5zSEVPO2Jjaeolg-flpYeFKRpNni3mgZcxgJvw2BtRLp6aQFQppfyDgthSzPAFeksnp2r_i6fDyAaCg="
)

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def evp_bytes_to_key(passphrase: str, salt: bytes, key_len: int = 32, iv_len: int = 16) -> tuple[bytes, bytes]:
    """CryptoJS OpenSSL-format EVP_BytesToKey (MD5, 1 iteration)."""
    d = b""
    prev = b""
    passphrase_bytes = passphrase.encode("utf8")
    while len(d) < key_len + iv_len:
        prev = hashlib.md5(prev + passphrase_bytes + salt).digest()
        d += prev
    return d[:key_len], d[key_len : key_len + iv_len]


def _pkcs7_pad(data: bytes) -> bytes:
    pad_len = 16 - (len(data) % 16)
    return data + bytes([pad_len]) * pad_len


def _pkcs7_unpad(data: bytes) -> bytes:
    if not data:
        return data
    pad_len = data[-1]
    if pad_len < 1 or pad_len > 16:
        raise CpcbError("bad PKCS7 padding (wrong key?)")
    return data[:-pad_len]


def encrypt(text: str, passphrase: str = SK) -> str:
    """Equivalent of CryptoJS AES.encrypt(text, passphrase).toString()."""
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    salt = _os.urandom(8)
    key, iv = evp_bytes_to_key(passphrase, salt)
    enc = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
    ct = enc.update(_pkcs7_pad(text.encode("utf8"))) + enc.finalize()
    return base64.b64encode(b"Salted__" + salt + ct).decode("ascii")


def decrypt(b64_text: str, passphrase: str = SK) -> str:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    raw = base64.b64decode(b64_text.strip())
    if raw[:8] != b"Salted__":
        raise CpcbError("not an OpenSSL Salted__ payload")
    salt, ct = raw[8:16], raw[16:]
    key, iv = evp_bytes_to_key(passphrase, salt)
    dec = Cipher(algorithms.AES(key), modes.CBC(iv)).decryptor()
    out = dec.update(ct) + dec.finalize()
    return _pkcs7_unpad(out).decode("utf8")


class CpcbError(RuntimeError):
    pass


class CpcbTokenSource:
    """Graceful token loader."""

    @staticmethod
    def from_capture(capture_path: str) -> str:
        cap = json.load(open(capture_path, encoding="utf8"))
        for capv in cap.values():
            post = capv.get("post")
            if not post:
                continue
            try:
                env = json.loads(decrypt(post))
            except Exception:  # noqa: BLE001
                continue
            auth = (env.get("header") or {}).get("Authorization")
            if auth:
                return auth
        raise CpcbError("no Authorization token in capture")


class CpcbClient:
    """Thin wrapper that mirrors the SPA's apiCaller() flow."""

    def __init__(self, token: str | None = None, timeout: float = 60.0, token_store: str | None = None):
        self.token = token or ""
        self.refresh_token = ""
        self.timeout = timeout
        self.token_store = token_store
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _UA})

    def _post_transport(self, body: str) -> requests.Response:
        return self._session.post(
            SERVICE_URL,
            data=body.encode("ascii"),
            headers={
                HEADER_NAME: HEADER_VALUE,
                "Content-Type": "text/plain",
                "Origin": "https://airquality.cpcb.gov.in",
                "Referer": "https://airquality.cpcb.gov.in/ccr/",
                "Accept": "application/json, text/plain, */*",
            },
            timeout=self.timeout,
        )

    # --- session bootstrap ------------------------------------------------
    def bootstrap_token(self) -> str:
        """Mint a fresh session token via the portal's default credentials.

        Mirrors the SPA's getTokenFromDefaultCredentials() flow.  Two routes
        verified in October 2026:
        1. direct  - POST the oauth endpoint on airquality.cpcb.gov.in itself
           (plain JSON response, no encryption; /oauth/token 405s GET, so it is
           POST-only).
        2. proxy   - the SPA's own transport: envelope encrypted with 'iv'
           submitted to service.php, response decrypted with 'iv'.  Used when
           route 1 is blocked.
        Saves token/refresh_token/expiry to ``token_store`` when set, so
        subsequent runs start with a live token.
        """
        oauth_url = (
            BASE + OAUTH_TOKEN_PATH + OAUTH_CLIENT_ID + "&client_secret=" + OAUTH_CLIENT_SECRET
        )
        data: dict[str, Any] | None = None
        try:
            resp = self._session.post(
                oauth_url,
                data=b"",
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except ValueError:
                    data = None
        except requests.RequestException:
            data = None
        if not isinstance(data, dict) or not data.get("access_token"):
            # SPA transport: envelope encrypted with 'iv', response decrypted with 'iv'
            envelope = {"methodType": "POST", "url": oauth_url, "payload": "{}", "enableEncryption": True}
            body = encrypt(json.dumps(envelope).strip(), HEADER_VALUE)
            resp = self._post_transport(body)
            if resp.status_code != 200:
                raise CpcbError(f"bootstrap HTTP {resp.status_code}: {resp.text[:160]}")
            txt = resp.text.strip()
            data = json.loads(txt) if (txt.startswith("{") or txt.startswith("[")) else json.loads(
                decrypt(txt, HEADER_VALUE)
            )
        if not isinstance(data, dict) or not data.get("access_token"):
            raise CpcbError(f"bootstrap: no access_token in response: {str(data)[:160]}")
        self.token = str(data["access_token"])
        self.refresh_token = str(data.get("refresh_token") or "")
        expires_in = int(data.get("expires_in") or 0)
        if self.token_store:
            try:
                store = Path(self.token_store)
                store.parent.mkdir(parents=True, exist_ok=True)
                store.write_text(
                    json.dumps(
                        {
                            "token": self.token,
                            "refresh_token": self.refresh_token,
                            "expires_in": expires_in,
                            "minted_at": datetime.now(timezone.utc).isoformat(),
                            "bootstrap": "default_credentials",
                        },
                        indent=1,
                    ),
                    encoding="utf8",
                )
            except OSError:
                pass
        return self.token

    # --- transport ------------------------------------------------------
    def call(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        token: str | None = None,
        _retries_left: int = 1,
    ) -> Any:
        auth = token if token is not None else self.token
        if not auth:
            self.bootstrap_token()
            auth = self.token
        envelope: dict[str, Any] = {
            "methodType": method.upper(),
            "url": url,
            "header": {"Authorization": auth, "Content-Type": "application/json"},
            "enableEncryption": True,
        }
        if payload is not None:
            envelope["payload"] = json.dumps(payload)
        body = encrypt(json.dumps(envelope).strip())
        resp = self._post_transport(body)
        if resp.status_code in (401, 403):
            # session token expired/invalidated; mint a fresh one once and retry
            if _retries_left > 0:
                self.bootstrap_token()
                return self.call(method, url, payload, token=None, _retries_left=_retries_left - 1)
            raise CpcbError(f"HTTP {resp.status_code} after re-bootstrap: {resp.text[:120]}")
        if resp.status_code != 200:
            raise CpcbError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        txt = resp.text.strip()
        if txt.startswith("{") or txt.startswith("[") or txt.startswith("<"):
            return json.loads(txt)
        return json.loads(decrypt(txt))

    def get(self, url: str, token: str | None = None) -> Any:
        return self.call("GET", url, token=token)

    def post(self, url: str, payload: dict[str, Any], token: str | None = None) -> Any:
        return self.call("POST", url, payload=payload, token=token)

    # --- data views ------------------------------------------------------
    def colour_ranges(self) -> Any:
        return self.get(QUERY + "get-colour-range-master/data?page=-1&perPage=-1")

    def city_avg_snapshot(self, city_id: str | None = None) -> Any:
        fq = f"city_id IS '{city_id}'" if city_id else ""
        return self.get(
            QUERY + "get-city-wise-avg-aqi-pm-data/data?page=-1&perPage=-1&filterQuery=" + fq
        )

    def city_hourly_avg(self, city_id: str) -> Any:
        return self.get(
            QUERY
            + "get-hourly-aqi-data-avg-view-v1/data?page=-1&perPage=-1&filterQuery=city_id IS '"
            + city_id
            + "'"
        )

    def city_last_3_days(self, city_id: str) -> Any:
        payload = {
            "complexData": {
                "get-city-wise-last-3-days-aqi": [
                    {"p_city_code": city_id, "p_lat": "", "p_long": ""}
                ]
            }
        }
        return self.post(
            QUERY + "get-city-wise-last-3-days-aqi/execute-procedure",
            payload,
        )
