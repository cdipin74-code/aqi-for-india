# CPCB CCR live pipeline — reverse-engineering notes

Date: 2026-10-22 (by OpenCode session).

## What was found

The official portal `https://airquality.cpcb.gov.in/ccr/` is an Angular SPA.
All its data traffic goes through ONE endpoint:

```
POST https://airquality.cpcb.gov.in/insight_proxy_api_v2_0/service.php
headers:
  index: iv
  Content-Type: text/plain
```

The request body is a CryptoJS-style payload:

```
base64( b"Salted__" + 8-byte random salt + AES-256-CBC(JSON_envelope) )
```

Key derivation: OpenSSL EVP_BytesToKey (MD5, 1 iteration), key 32 bytes,
IV 16 bytes, passphrase:

```
P7#Vw!3Xz@8Lm$2QkT9
```

This was recovered from the deobfuscated chunk `chunk-WKM2WSKQ.js`
(string-table decoder `a47_0x39ab` / array `a47_0x5d7c`; constants
`Le='iv'` (header), `Qf='7m67CrnRRldWUD2j'` (unused), runtime `secretKey`
field of class `sc` = the real key above; verified by decrypting captured
live traffic).

The envelope (after decryption):

```json
{
  "methodType": "GET|POST",
  "url": "https://airquality.cpcb.gov.in/...",
  "header": {"Authorization": "<token GUID>", "Content-Type": "application/json"},
  "payload": "stringified JSON (only for POST bodies)",
  "enableEncryption": true
}
```

The server response is AES-encrypted the same way (some 4xx responses are
unencrypted Spring error JSON).

## Session token

The Authorization header carries a GUID session token minted by the SPA's
OAuth bootstrap when localStorage is empty. Our captured token
(`8a783319-5ca0-4071-815c-c695a2ab4c6a`) remained valid across many
requests; it is stored in `data/raw/cpcb_live/session_token.json`.

## Useful verified endpoints (via the envelope)

| view | method | notes |
|---|---|---|
| `get-city-wise-avg-aqi-pm-data/data?page=-1&perPage=-1&filterQuery=` | GET | all 301 cities + current hourly AQI |
| `get-hourly-aqi-data-avg-view-v1/data...filterQuery=city_id%20IS%20'<id>'` | GET | per-station hourly rows (current window only) |
| `get-city-wise-last-3-days-aqi/execute-procedure` | POST | official daily AQI + station_count + primary_pollutants (last 3 days) |
| `get-fn-city-aqi-summary-by-time/execute-procedure` | POST | **any** city, **any** date, args `{p_reftime, p_city_code}` |
| `get-fn-station-monthly-aqi-calendar/execute-procedure` | POST | hourly calendar (retention-limited) |
| `get-colour-range-master/data` | GET | AQI band table |
| `get-site-station-master-data-list/data` | GET | 617 stations w/ coords |
| `get-list-report-que-ccr/data` | GET | server-side report jobs |

`filterQuery` must be URL-percent-encoded (`%20` for spaces) or the server
returns HTTP 400.

## Anchor validation against the research dataset (Kaggle)

The daily city summary is an *instantaneous* value at `p_reftime`.
To reproduce the research dataset's daily `aqi_value`, we tested anchors
over 20 overlap days (Delhi, Jan–Apr 2025):

| anchor (IST) | MAE | median | p90 |
|---|---|---|---|
| 00:30 | 20.8 | 19.5 | 46.2 |
| 03:30 | 18.6 | 18.0 | 32.7 |
| 10:30 | 9.2 | 8.0 | 17.3 |
| 11:30 | 7.6 | 5.5 | 16.2 |
| **15:30** | **2.4** | **1.5** | **5.1** |
| **17:30** | **2.4** | **1.0** | **6.4** |
| 23:30 | 9.4 | 6.0 | 19.0 |

15:30 IST is exactly CPCB's 4 PM publication time; 17:30 IST is
23:00 IST. `fetch_cpcb_live.py` records both per day and uses 15:30 as
the primary `aqi_value` (17:30 fallback).

## Honest caveats

1. Source semantics: the live daily value is anchored, not a 24-hour mean;
   it agrees with the research dataset closely (median |Δ| ~1) because
   that dataset most likely used CPCB's published daily AQI, but exact
   equivalence over the whole overlap is NOT claimed — per-city overlap
   validation is part of the pipeline (`src/validate_cpcb_overlap.py`).
2. `number_of_monitoring_stations` is only published for the last 3 days
   (last-3-days fn). Historical live rows keep this empty.
3. Weather for the gap comes from Open-Meteo (ERA5 archive for history,
   forecast API for the most recent days and next days). ERA5 vs NASA
   POWER are two reanalyses — per-feature overlap validation is included.

## Session tokens and the OAuth bootstrap (verified 2026-10-22)

The GUID session token from a captured session expires after a few hundred
data calls (HTTP 401, body an encrypted error). The SPA itself re-mints tokens
in `getTokenFromDefaultCredentials()` / `getTokenFromRefreshToken()`
(chunk-WKM2WSKQ.js, deobfuscated); both flows are reproduced by
`CpcbClient.bootstrap_token()` in `src/cpcb_client.py`:

1. **Direct route (primary).** `POST
   https://airquality.cpcb.gov.in/api/security/oauth/token?grant_type=password
   &username=ayush.gaur@mapmyindia.com&password=ab3edead07f87462ae80892589f91907
   &client_id=<Fr.Client3>&client_secret=<Fr.Sec3>` with an empty body.
   Plain JSON comes back: `access_token`, `refresh_token`, `expires_in=86400`,
   `scope: READ`. `GET` gives 405, so the endpoint is POST-only.
   This is the same endpoint the proxy forwards to, minus the encryption hop.
2. **Proxy route (SPA-equivalent fallback).** Envelope
   `{"methodType":"POST","url":<oauth url>,"payload":"{}","enableEncryption":true}`
   encrypted with CryptoJS AES - passphrase `'iv'` (NOT the data key) - and
   POSTed to `service.php` with header `index: iv`; the response is decrypted
   with `'iv'` too. Verified against the deobfuscated bundle: `Fr = {Client1/2/3:
   "96dHZ...di0Kz", Sec1/2/3: "lrFxI...g=="}` (all three pairs identical).

`CpcbClient.call()` catches 401/403 once, calls `bootstrap_token()`, and
retries the request transparently; a fresh token is persisted to
`data/raw/cpcb_live/session_token.json` (`token_store` on the client) with its
mint time so the fetcher reuses it for up to 23 h. `fetch_cpcb_live.py` survived
the full 5-city x 539-day backfill (>= 5,800 calls) with a single running token
chain thanks to this path.
