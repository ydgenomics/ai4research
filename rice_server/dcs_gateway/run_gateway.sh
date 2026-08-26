#!/usr/bin/env bash
# 兼容 /bin/sh=dash（云平台 org_web:sanic 镜像）：不用 pipefail
set -eu
cd "$(dirname "$0")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
exec python app.py