Quick setup — DarkReaper Order Book Dashboard

Prereqs
- Python 3.10 or newer
- Git (optional)

1) Create and activate a virtual environment
PowerShell:

```powershell
python -m venv .env
& .\.env\Scripts\Activate.ps1
```

2) Install dependencies

```powershell
pip install -U pip
pip install -r user_data/requirements.txt
```

3) Run the dashboard (development)

```powershell
python user_data/dashboard.py
```

- The app will run on http://localhost:8050 by default
- To change refresh/settings, open the app and use the ⚙️ Settings panel; settings are persisted to `dashboard_symbols.json` in the repository root

Notes
- The code requires Python 3.10+ (uses `X | None` types)
- `ccxt` is used for public exchange REST calls; WebSocket trade collection uses the `websockets` package
- SQLite (`orderbook_cache.db`) is used for storing aggregated trade candles (standard library `sqlite3`)

Troubleshooting
- If Dash/Plotly import fails, ensure the installed `dash` version is recent (>=2.9). Installing `dash` pulls `plotly` automatically but `plotly` is included explicitly in `requirements.txt` as a safeguard.
- On Windows, run PowerShell as administrator only if you run into permission issues creating the venv

If you want, I can also pin exact versions from your project environment or add a `pyproject.toml` / `requirements-dev.txt` set for development.
