#!/bin/sh
# floortint 起動スクリプト。
# 1. サンプル案件の置き場（FLOORTINT_SAMPLES）があれば画面に「サンプル案件」を出す。無ければアップロードのみ
# 2. Streamlit を起動する
set -e

PORT="${PORT:-8000}"
export FLOORTINT_SAMPLES="${FLOORTINT_SAMPLES:-/app/samples}"
export FLOORTINT_OUT="${FLOORTINT_OUT:-/tmp/floortint_out}"
export FLOORTINT_LIBRARY="${FLOORTINT_LIBRARY:-/app/library}"
mkdir -p "$FLOORTINT_OUT" "$FLOORTINT_LIBRARY/materials" "$FLOORTINT_LIBRARY/projects" 2>/dev/null || true
if [ -w "$FLOORTINT_LIBRARY" ]; then
  echo "[entrypoint] library: $FLOORTINT_LIBRARY (床材写真と物件設定の保存先。永続ストレージを当てないと再起動で消える)"
else
  echo "[entrypoint] WARNING: library dir not writable: $FLOORTINT_LIBRARY"
fi

if ls "$FLOORTINT_SAMPLES"/*.pdf >/dev/null 2>&1; then
  echo "[entrypoint] samples: $FLOORTINT_SAMPLES"
else
  echo "[entrypoint] samples not found in $FLOORTINT_SAMPLES (upload only)"
fi

echo "[entrypoint] starting streamlit on 0.0.0.0:${PORT}"
exec streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port "${PORT}" \
  --server.headless true \
  --browser.gatherUsageStats false \
  --server.maxUploadSize "${MAX_UPLOAD_MB:-200}"
