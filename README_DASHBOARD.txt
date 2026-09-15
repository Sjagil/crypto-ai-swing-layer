Round44 Operations Dashboard

Install from repo root:
  unzip round44_ops_dashboard_bundle.zip -d /Users/ayoubalhari/Downloads/crypto-ai-swing-layer

Validate:
  cd /Users/ayoubalhari/Downloads/crypto-ai-swing-layer
  source .venv/bin/activate
  python -m pytest -q tests/test_web_dashboard.py

Run:
  python scripts/run_web_dashboard.py

Default URL:
  http://127.0.0.1:8765/

The dashboard is read-only. It exposes no POST/order endpoints and does not grant live authority.
