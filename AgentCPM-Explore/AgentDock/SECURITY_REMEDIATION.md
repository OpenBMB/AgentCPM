# AgentDock 安全修复说明

## 修复范围

本次修复针对两个已确认问题：

1. 管理端未认证创建节点，默认创建 `privileged` 容器并通过 Docker Socket 管理宿主机。
2. 节点文件读取工具对已存在的绝对路径绕过 `/app/data` 根目录，导致容器内任意文件读取。

同时修复了管理端凭据向节点转发、仓库中硬编码 API Key、节点请求日志记录凭据和若干部署配置不一致问题。

## 部署要求

```bash
cp .env.example .env
openssl rand -hex 32
openssl rand -hex 32
```

将两个不同的随机值分别填入：

- `AGENTDOCK_API_TOKEN`：仅管理端使用。
- `AGENTDOCK_NODE_API_TOKEN`：仅节点接口使用，管理端代理请求时注入。

然后从当前源码重新构建：

```bash
docker compose build --no-cache
docker compose up -d
```

不要把管理端 Token 放进节点环境。不要把旧 Docker Hub 镜像当作已修复版本；源码修复尚未发布为新的镜像标签。

## 现有环境处置

- 立即撤销并重新生成仓库、报告或日志中曾暴露过的搜索和模型 API Key。
- 停止并删除旧的 `privileged` AgentDock 节点。
- 检查 Docker 容器、宿主机文件和应用日志是否有异常访问。
- 在公网入口前增加 TLS、访问控制和限流；本次 Token 认证不是租户隔离方案。

## 验证边界

已完成 Python 编译、三套 Compose 配置解析、认证边界检查和文件路径边界检查；未执行附件中的逃逸 PoC，也未在完整生产镜像中重建并进行宿主机逃逸回归。源码中的代码执行器仍应按不可信代码处理，容器安全配置不能替代沙箱。
