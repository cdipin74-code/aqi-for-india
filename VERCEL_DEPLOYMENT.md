# Deploying India AQI & Early Warning System to Vercel

This repository is pre-configured for **zero-configuration serverless deployment on Vercel**.

---

## What is Included for Vercel:

1. **`public/index.html`**: Modern, responsive web application (Tailwind CSS, Glassmorphic UI) with:
   - State selector (32 Indian States / UTs)
   - City selector (292 monitored Indian cities, dynamically filtered)
   - Date selector with one-click presets
   - One-click copy for the formatted output block
2. **`api/index.py`**: High-performance Python serverless function:
   - Zero heavy dependencies (uses Python built-in standard library `http.server` & `sqlite3`)
   - Sub-millisecond cold start
   - Serves `/api?city=Patna&state=Bihar&date=2024-11-15`
3. **`api/aqi_light.db`**: Pre-indexed compact database covering all 292 cities and active stations.
4. **`vercel.json`**: Routing and configuration rules.

---

## Deployment Steps

### Option A: Deploy via Vercel Web Dashboard (Recommended - 1 Click)

1. Go to [vercel.com](https://vercel.com) and log in with your GitHub account (`cdipin74-code`).
2. Click **"Add New..."** → **"Project"**.
3. Select your repository: **`cdipin74-code/aqi-for-india`**.
4. Leave all settings at default:
   - **Framework Preset**: *Other*
   - **Root Directory**: `./`
5. Click **"Deploy"**.
6. In ~30 seconds, Vercel will give you a live production URL (e.g. `https://aqi-for-india.vercel.app`).

---

### Option B: Deploy via Vercel CLI

If you have Node.js installed, open PowerShell in this directory and run:

```powershell
npx vercel
```

Follow the prompts (accept default options). To deploy to production:

```powershell
npx vercel --prod
```

---

## Live API Endpoints

Once deployed, your live site also serves an instant REST API:

* **Get Report**:
  ```http
  GET /api?city=Patna&state=Bihar&date=2024-11-15
  ```
* **Get All 292 Cities by State**:
  ```http
  GET /cities.json
  ```
