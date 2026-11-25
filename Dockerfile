# floortint v3.1 — 平面図（＋床仕上図 or 指示書 PDF/PPTX/手書き）→ 平面着色（床材の模様を自動で塗る）Streamlit アプリ
# Azure App Service (Web App for Containers, Linux) 想定。GPU・外部 API は不要。
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    FLOORTINT_SAMPLES=/app/samples \
    FLOORTINT_OUT=/tmp/floortint_out \
    FLOORTINT_LIBRARY=/app/library

WORKDIR /app

# 室名の描画に使う日本語フォント（IPA P ゴシック）。無いと室名が豆腐になる
# LibreOffice（impress）は PPTX の指示書を PDF に変換するために使う（約 400MB。PPTX を使わないなら外してよい）
RUN apt-get update \
 && apt-get install -y --no-install-recommends fonts-ipafont-gothic libreoffice-impress libreoffice-core \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 非 root 実行
RUN useradd -m appuser && mkdir -p /app/samples /app/library /tmp/floortint_out && chown -R appuser /app /tmp/floortint_out
USER appuser

# App Service は WEBSITES_PORT=8000 でこのポートに転送する
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/_stcore/health',timeout=3)" || exit 1

CMD ["/entrypoint.sh"]
