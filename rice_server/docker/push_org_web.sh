#!/usr/bin/env bash
# ============================================================================
# push_org_web.sh — 将 org_web 镜像推送到 Docker Hub (ydgenomics)
#
# 用法:
#   ./push_org_web.sh              # 推送全部标签 (latest + jupyter)
#   ./push_org_web.sh latest       # 只推送 latest
#   ./push_org_web.sh jupyter      # 只推送 jupyter
#
# 注意:
#   - 需先登录: docker login -u ydgenomics (密码在 https://hub.docker.com)
#   - 镜像共 ~30 GB, 推送耗时取决于上行带宽 (建议后台执行: nohup ./push_org_web.sh &)
# ============================================================================
set -euo pipefail

NAMESPACE="ydgenomics"
IMAGE="org_web"

# 未登录则提示
if ! docker system info 2>/dev/null | grep -q "Username: $NAMESPACE"; then
    echo "[ERROR] 未登录 Docker Hub 账号 $NAMESPACE, 请先执行:"
    echo "  docker login -u $NAMESPACE"
    exit 1
fi

# 默认推送全部标签
TAGS=("$@")
if [[ ${#TAGS[@]} -eq 0 ]]; then
    TAGS=("latest" "jupyter")
fi

for tag in "${TAGS[@]}"; do
    echo "============================================================"
    echo "[1/2] 打标签: ${IMAGE}:${tag} -> ${NAMESPACE}/${IMAGE}:${tag}"
    docker tag "${IMAGE}:${tag}" "${NAMESPACE}/${IMAGE}:${tag}"

    echo "[2/2] 推送: ${NAMESPACE}/${IMAGE}:${tag}"
    docker push "${NAMESPACE}/${IMAGE}:${tag}"
done

echo "============================================================"
echo "✅ 全部推送完成。查看: https://hub.docker.com/r/${NAMESPACE}/${IMAGE}"
