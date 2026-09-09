#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os
from pathlib import Path

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--packet",required=True); p.add_argument("--output",required=True); p.add_argument("--crypto-root",default="/Users/ayoubalhari/Downloads/crypto"); p.add_argument("--refresh-external",action="store_true"); a=p.parse_args(); root=Path(__file__).resolve().parents[1]; os.environ["CRYPTO_REPO_PATH"]=str(Path(a.crypto_root).expanduser().resolve())
    from crypto_ai_swing.integrations.whole_context import enrich_packet
    from crypto_ai_swing.settings import Settings
    result=enrich_packet(json.loads(Path(a.packet).read_text(encoding="utf-8")),Settings.load(root),refresh_external=a.refresh_external); out=Path(a.output).expanduser().resolve(); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8"); print(json.dumps(result,indent=2,sort_keys=True,default=str)); return 0
if __name__=="__main__": raise SystemExit(main())
