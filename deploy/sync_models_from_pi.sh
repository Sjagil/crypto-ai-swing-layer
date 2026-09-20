#!/bin/bash
set -euo pipefail

HOST="${PI_HOST:-pi@192.168.40.51}"
REMOTE="${PI_SWING_OUTPUT:-/mnt/data/crypto-ai-swing/output/crypto_ai_swing}"
LOCAL_SWING="${LOCAL_SWING:-/Users/ayoubalhari/Downloads/crypto-ai-swing-layer}"
LOCAL="$LOCAL_SWING/output/crypto_ai_swing"
REMOTE_PREFIX="/home/pi/sjagil/crypto-ai-swing-layer"

mkdir -p "$LOCAL/agents" "$LOCAL/research" "$LOCAL/pi_learning_worker" "$LOCAL/learning_worker"
LOCKDIR="$LOCAL/.pi_model_sync.lockdir"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "MODEL_SYNC_ALREADY_RUNNING"
  exit 0
fi
trap 'rm -rf "$LOCKDIR"' EXIT

# Immutable artifacts first; pointers are deliberately excluded until the files
# they reference are present locally.
rsync -az --delete-delay \
  --exclude='*.pointer.json' \
  --exclude='state.json' \
  "$HOST:$REMOTE/agents/" "$LOCAL/agents/"

rsync -az \
  "$HOST:$REMOTE/research/" "$LOCAL/research/" 2>/dev/null || true
rsync -az \
  "$HOST:$REMOTE/pi_learning_worker/" "$LOCAL/pi_learning_worker/" 2>/dev/null || true
rsync -az \
  "$HOST:$REMOTE/learning_worker/" "$LOCAL/learning_worker/" 2>/dev/null || true

TMP="$(mktemp -d "$LOCAL/.pi-pointers.XXXXXX")"
trap 'rm -rf "$TMP" "$LOCKDIR"' EXIT
rsync -az --include='*/' --include='*.pointer.json' --exclude='*' \
  "$HOST:$REMOTE/agents/" "$TMP/"

python3 - "$TMP" "$LOCAL/agents" "$REMOTE_PREFIX" "$LOCAL_SWING" <<'PY'
from __future__ import annotations
import hashlib,json,os,sys
from pathlib import Path
src=Path(sys.argv[1]); dst=Path(sys.argv[2]); old=sys.argv[3]; new=sys.argv[4]

def rewrite(value):
    if isinstance(value,str):
        return new + value[len(old):] if value.startswith(old) else value
    if isinstance(value,list): return [rewrite(x) for x in value]
    if isinstance(value,dict): return {k:rewrite(v) for k,v in value.items()}
    return value

def sha(path:Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

count=0
for path in src.rglob('*.pointer.json'):
    payload=rewrite(json.loads(path.read_text()))
    artifact=payload.get('artifact_path')
    expected=payload.get('artifact_hash') or payload.get('source_artifact_hash')
    if artifact:
        ap=Path(artifact)
        if not ap.is_file():
            raise SystemExit(f'POINTER_ARTIFACT_MISSING:{path.name}:{ap}')
        if expected and sha(ap) != str(expected):
            raise SystemExit(f'POINTER_HASH_MISMATCH:{path.name}:{ap}')
    rel=path.relative_to(src); target=dst/rel
    target.parent.mkdir(parents=True,exist_ok=True)
    tmp=target.with_suffix(target.suffix+'.tmp')
    tmp.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+'\n')
    os.replace(tmp,target); count+=1
print(f'MODEL_POINTERS_INSTALLED={count}')
PY

echo "PI_MODEL_SYNC=PASS"
