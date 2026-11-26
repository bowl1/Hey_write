可以。**先手动部署 Azure Container Apps，Supabase 和 Vercel 保留原样。**按下面顺序做。

**1．准备一个能被 Azure 拉取的 Docker 镜像**

如果你的镜像已经在 Docker Hub、GHCR 或 Azure Container Registry，直接用现有地址。

如果镜像只在 EC2 或电脑里，需要先推送到镜像仓库。比如使用 Docker Hub，在 Dockerfile 所在目录运行：

```bash
docker login

docker buildx build \
  --platform linux/amd64 \
  -t 你的DockerHub用户名/heywrite-backend:azure-v1 \
  --push .
```

这里的用户名需要替换。**先确认 `.dockerignore` 排除了 `.env`、密钥和其他敏感文件**，再构建；应用需要的密钥之后在 Azure 配置。

**2．在 Azure Portal 创建 Container App**

搜索 **Container Apps → Create → Container App**，填写：

| 配置 | 选择 |
|---|---|
| Subscription | 你的订阅 |
| Resource group | 新建 `rg-heywrite-azure` |
| Container app name | `heywrite-backend` |
| Deployment source | **Container image** |
| Region | 例如 North Europe，以可用配额为准 |
| Environment | 新建 |
| Workload profile | **Consumption** |

**3．填写镜像信息**

取消勾选 **Use quickstart image**。

使用 Docker Hub 的话：

| 配置 | 示例 |
|---|---|
| Image source | Docker Hub or other registries |
| Registry server | `docker.io` |
| Image | `你的用户名/heywrite-backend` |
| Tag | `azure-v1` |

私有镜像需要填写仓库认证信息；公开镜像通常不用。

**4．配置运行环境**

把当前 EC2 后端使用的必要环境变量配置到 Azure，包括数据库、认证、模型 API 和 tracing 配置，**变量名沿用你代码里的名字**。

- 密钥先加入 **Secrets**。
- 在容器的 **Environment variables** 中引用对应 Secret。
- 普通配置可以直接填值。
- 本地 `.env` 文件不会自动出现在 Azure 中。

先沿用 Docker 镜像已有的启动命令，不必额外覆盖。

**5．配置公网入口和端口**

在 **Ingress** 中：

- 开启 Ingress。
- 选择 **Accepting traffic from anywhere**。
- 使用 HTTP 入口，关闭允许不安全 HTTP 的选项，保留 HTTPS。
- **Target port 必须与 FastAPI 实际监听端口一致**。

例如，只有你的启动命令是下面这样时，才填写 `8000`：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

`main:app` 只是示例，实际沿用你项目的入口。**应用必须监听 `0.0.0.0`，不能只监听 `127.0.0.1`。**

以上创建入口和字段可参考[微软官方步骤](https://learn.microsoft.com/en-us/azure/container-apps/get-started-existing-container-image-portal)。

**6．设置资源并创建**

初次可以尝试：

- CPU：**0.5 vCPU**
- 内存：**1 GiB**
- 最小副本：**0**
- 最大副本：**1**

这只是起始配置，是否够用要看实际内存和启动表现。创建页面没有缩放选项的话，部署后再修改。

点击 **Review + create**，检查后创建。

**7．先测试 Azure 后端**

部署完成后打开 **Application URL**，检查：

- 健康检查接口能否正常返回。
- 如果启用了 FastAPI 文档，访问 `/docs`。
- 日志里是否有缺失环境变量、数据库连接或启动错误。
- 模板检索、生成和会话保存是否正常。

根路径 `/` 返回404不一定是部署失败，可能只是你没有定义这个路由。

**8．最后连接 Vercel**

Azure 后端确认正常后：

1. 在后端 CORS 中允许你的 Vercel 域名。
2. 修改 Vercel 项目的后端 API 地址。
3. 重新部署前端。
4. 完整测试登录、生成、多轮编辑和会话恢复。

**旧 EC2 先保留，等 Azure 版本通过完整测试后再停。你现在第一步只需要确认：现有 HeyWrite 镜像是否已经在镜像仓库里。**

## GitHub Actions 自动发布

仓库中的 `.github/workflows/deploy-backend.yml` 在测试和数据库迁移通过后，
调用 `scripts/deploy-azure.sh`。脚本在 GitHub Ubuntu runner 上使用 Docker 构建
`linux/amd64` 镜像，通过 OIDC 身份登录 ACR 并推送，再更新 Container App。
它不调用 `az acr build`，因此不依赖 ACR Tasks；手动运行脚本时需要启动本地 Docker。

GitHub repository variables 配置 `BACKEND_DEPLOY_TARGET=azure`、
`AZURE_CONTAINER_APP`、`AZURE_RESOURCE_GROUP`、`AZURE_CONTAINER_REGISTRY`；
repository secrets 配置 `AZURE_CLIENT_ID`、`AZURE_TENANT_ID`、
`AZURE_SUBSCRIPTION_ID`，并保留数据库迁移用的 `DATABASE_ADMIN_URL`。
部署身份需要应用更新权限和 ACR 推送权限。发布后脚本检查 revision 就绪、
健康接口、公开模板和认证保护，失败时尝试从上一就绪 revision 复制配置回退。
