#!/usr/bin/env bash
# 将后端发布到已经创建并配置好的 Azure Container App。
# 前置条件：已通过 az login 或 GitHub OIDC 登录，且拥有 ACR 构建和应用更新权限。
# 运行密钥和环境变量由 Azure 保存，本脚本不读取本地 .env。
# GitHub Actions 在测试、数据库迁移通过后调用；本脚本本身不执行迁移。
# 手动调用示例：bash scripts/deploy-azure.sh "$(git rev-parse HEAD)"
# 命令失败、未定义变量或管道中的命令失败时退出。
set -euo pipefail

# 第一个参数必须是完整的 40 位 Git commit SHA，用作镜像标签。
revision=${1:?Pass the Git commit SHA}
[[ "$revision" =~ ^[0-9a-f]{40}$ ]] || exit 2

# 这三个环境变量由 GitHub Actions 的 repository variables 传入；手动运行时先 export。
# AZURE_CONTAINER_REGISTRY 填注册表名称，不带 .azurecr.io 域名。
: "${AZURE_RESOURCE_GROUP:?Set AZURE_RESOURCE_GROUP}"
: "${AZURE_CONTAINER_APP:?Set AZURE_CONTAINER_APP}"
: "${AZURE_CONTAINER_REGISTRY:?Set AZURE_CONTAINER_REGISTRY (registry name)}"

# 固定在仓库根目录运行，确保 Docker 构建上下文路径正确。
cd "$(dirname "$0")/.."
app_args=(--name "$AZURE_CONTAINER_APP" --resource-group "$AZURE_RESOURCE_GROUP")

# 本脚本只支持 Single 模式：Azure 在新 revision 就绪后自动切换流量。
mode=$(az containerapp show "${app_args[@]}" --query properties.configuration.activeRevisionsMode -o tsv)
[[ "$mode" = Single ]] || { echo 'Expected single revision mode'; exit 1; }

# 保存上一版就绪 revision，作为后续失败回退的配置来源。
# 首次创建应用需先手动完成，本脚本要求已经存在就绪版本。
previous=$(az containerapp show "${app_args[@]}" --query properties.latestReadyRevisionName -o tsv)
test -n "$previous"

# 查询 ACR 的实际登录域名，拼接带 commit SHA 的镜像地址。
registry=$(az acr show --name "$AZURE_CONTAINER_REGISTRY" --query loginServer -o tsv)
image="$registry/heywrite-backend:$revision"

# 使用 ACR 云端构建并推送镜像，无需本地 Docker daemon。
# 只上传 chatbox-backend 目录，.dockerignore 排除 dotenv 等文件。
# 构建失败时还未修改应用，不需要触发回退。
az acr build --registry "$AZURE_CONTAINER_REGISTRY" --image "heywrite-backend:$revision" \
  --file dockerfile chatbox-backend --only-show-errors --output none

# revision 名称结合 commit、Actions run ID 和重试次数，减少重复发布时的命名冲突。
# 手动运行时用当前时间代替 run ID；Azure revision 的完整名称为“应用名--后缀”。
suffix="git-${revision:0:12}-${GITHUB_RUN_ID:-$(date +%s)}-${GITHUB_RUN_ATTEMPT:-1}"
candidate="$AZURE_CONTAINER_APP--$suffix"

# 从旧 revision 复制容器配置，创建一个新的回退 revision。
# 回退不会撤销数据库迁移，也不会恢复应用级共享 Secret 的历史值。
# 无论回退是否成功，都保留原部署的失败退出码，让 Actions 标记失败。
rollback() {
  result=$?
  if [ "$result" -ne 0 ]; then
    echo "Deployment failed; restoring configuration from $previous"
    az containerapp revision copy "${app_args[@]}" --from-revision "$previous" \
      --revision-suffix "rollback-$(date +%s)" --output none || \
      echo "ERROR: rollback failed; restore $previous in the Azure portal"
  fi
  exit "$result"
}

# 从更新应用开始，任意命令失败或检查不通过，退出时都会尝试回退。
trap rollback EXIT

# 更新镜像并创建 revision，保留已有的环境变量、Secret 引用和其他应用配置。
az containerapp update "${app_args[@]}" --image "$image" \
  --revision-suffix "$suffix" --output none

# 最多轮询 60 次，间隔 5 秒（总耗时还包含每次 Azure CLI 请求）。
# 必须确认本次 candidate 就绪，不能把旧版本仍在正常运行误判为部署成功。
ready=false
for ((attempt=0; attempt<60; attempt++)); do
  serving=$(az containerapp show "${app_args[@]}" --query properties.latestReadyRevisionName -o tsv)
  if [ "$serving" = "$candidate" ]; then ready=true; break; fi
  sleep 5
done
[ "$ready" = true ] || { echo 'Candidate did not become ready'; exit 1; }

# 通过应用的公开 HTTPS 域名检查实际服务响应；这里不经过 Vercel。
# /health 检查进程响应；/public/templates 检查公开模板读取。
# 这些检查不能替代真实登录后的数据库/RLS 和模型生成验证。
fqdn=$(az containerapp show "${app_args[@]}" --query properties.configuration.ingress.fqdn -o tsv)
test -n "$fqdn"
curl -fsS --retry 5 --retry-delay 2 --max-time 20 "https://$fqdn/health" >/dev/null
curl -fsS --retry 3 --retry-delay 2 --max-time 20 "https://$fqdn/public/templates" >/dev/null

# 私有接口在未携带登录 token 时应返回 401，验证认证保护没有意外失效。
status=$(curl -sS --max-time 20 -o /dev/null -w '%{http_code}' "https://$fqdn/agent/memory")
[ "$status" = 401 ] || { echo 'Authentication smoke check failed'; exit 1; }

# 所有检查通过，移除退出回退钩子，正常结束发布。
trap - EXIT
echo "Azure deployment verified: $candidate at https://$fqdn"
