# AgentDock

<p align="center">
  【<a href="README_zh.md">中文</a> | English】
</p>

<p align="center">
  <strong>Dynamic MCP Container Orchestration Platform</strong>
</p>

<p align="center">
  A Docker-based container orchestration system for deploying and managing<br>
  MCP (Model Context Protocol) services with dynamic lifecycle management.
</p>

---

## ✨ Features

- 🐳 **Dynamic Container Management** - Auto create, start, stop, and remove Docker containers
- 📊 **Resource Monitoring** - Real-time CPU, memory, and disk usage monitoring
- 🔄 **Health Checks** - Automatic container health detection with auto-restart
- 🌐 **MCP Routing** - Streamable-HTTP MCP protocol request routing
- 📈 **Web Dashboard** - Visual container management interface

## 🏗️ Architecture

```
AgentDock/
├── master/                     # Manager service
│   ├── main.py                 # FastAPI main application
│   ├── config.py               # Configuration management
│   ├── node.py                 # Node management routes
│   ├── config.yml              # Default configuration
│   ├── dockerfile              # Docker build file
│   └── templates/              # Web templates
├── node/                       # Base node image
├── agentdock-node-full/        # Full-feature MCP node
├── agentdock-node-explore/     # Explore MCP node (search & analysis)
├── docker-compose.yml          # Docker Compose configuration
└── .env.example                # Environment variables template
```

## 🚀 Quick Start

### Security configuration (required)

This version rejects unauthenticated manager and node tool requests. Set two different strong random values in `.env`:

```bash
AGENTDOCK_API_TOKEN=$(openssl rand -hex 32)
AGENTDOCK_NODE_API_TOKEN=$(openssl rand -hex 32)
```

The manager token is only for manager APIs; the node token is only injected by the manager into nodes. Rebuild all services from this source checkout instead of continuing to use old Docker Hub node images. Any search or model API keys previously exposed in the repository must be revoked and regenerated with their providers.

### Option 1: Pull from Docker Hub (legacy images only)

The old Docker Hub Manager/Node images do not contain these security fixes. Do not use `docker-compose-hub.yml` as the production remediation; use the source build below.

The fastest way to get started - no build required!

**1. Configure Environment**

```bash
# Clone the repository (or just download the compose file)
git clone https://github.com/OpenBMB/AgentCPM.git
cd AgentCPM/AgentCPM-Explore/AgentDock

# Create environment file
cp .env.example .env
# Edit .env file with your MongoDB credentials
```

**2. Pull and Start Services**

```bash
# Pull images from Docker Hub
docker compose -f docker-compose-hub.yml pull

# Start all services
docker compose -f docker-compose-hub.yml up -d
```

**3. Access Dashboard**

Open browser: `http://localhost:8080`, then sign in with `AGENTDOCK_API_TOKEN`.

#### Docker Hub Images

| Image | Description | Architectures |
|-------|-------------|---------------|
| `sailaoda/agentdock-manager` | Main orchestration service | amd64, arm64 |
| `sailaoda/agentdock-node-full` | Full-feature MCP server | amd64, arm64 |
| `sailaoda/agentdock-node-explore` | Explore MCP server | amd64, arm64 |

```bash
# Pull individual images
docker pull sailaoda/agentdock-manager:latest
docker pull sailaoda/agentdock-node-full:latest
docker pull sailaoda/agentdock-node-explore:latest
```

---

### Option 2: Build from Source

Build images locally from source code.

#### 🌍 Global Build (Default)

For users **outside China**, use the default Dockerfiles with official mirrors:

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env file with your MongoDB credentials

# 2. Build and start services
docker compose build
docker compose up -d
```

#### 🇨🇳 China Build (Aliyun Mirrors)

For users **in China**, use the `.cn` Dockerfiles with Aliyun mirrors for faster builds:

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env file with your MongoDB credentials

# 2. Build and start services (using China mirrors)
docker compose -f docker-compose.cn.yml build
docker compose -f docker-compose.cn.yml up -d
```

**Files for China Build:**
- `master/dockerfile.cn`
- `agentdock-node-full/dockerfile.cn`
- `agentdock-node-explore/Dockerfile.cn`
- `docker-compose.cn.yml`

**3. Access Dashboard**

Open browser: `http://localhost:8080`

## 📦 Services

| Service | Description | Ports |
|---------|-------------|-------|
| `agentdock-manager` | Main orchestration dashboard | 8080 |
| `agentdock-mongodb` | Database for node persistence | 27017 |
| `agentdock-node-full` | Full-feature MCP server | 8004, 8092 |
| `agentdock-node-explore` | Explore MCP server (search & analysis) | 8014, 8102 |

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `MONGODB_USERNAME` | MongoDB username | ✅ |
| `MONGODB_PASSWORD` | MongoDB password | ✅ |
| `AGENTDOCK_API_TOKEN` | Manager API token | ✅ |
| `AGENTDOCK_NODE_API_TOKEN` | Node API token; must differ from manager token | ✅ |
| `READ_FILE_LLM_API_KEY` | File reader LLM API key | ❌ |
| `READ_FILE_LLM_BASE_URL` | File reader LLM base URL | ❌ |
| `JINA_API_KEY` | Jina Reader API key | ❌ |
| `GOOGLE_SERP_API_KEY` | Google SERP API key | ❌ |

### Resource Limits

Default resource allocation:

- **agentdock-manager**: 2 cores / 4GB
- **agentdock-mongodb**: 2 cores / 6GB
- **agentdock-node-full**: 8 cores / 32GB
- **agentdock-node-search**: 4 cores / 16GB

Adjust in `docker-compose.yml` as needed.

## 🔌 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web dashboard |
| `/alive` | GET | Health check |
| `/node/create` | POST | Create new node |
| `/node/close` | POST | Stop node |
| `/node/release` | POST | Release node |
| `/node/details` | GET | Get node details |
| `/container/{id}/mcp` | POST | MCP request routing |

### Resource Monitoring

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/resources/system` | GET | System resources |
| `/api/resources/docker` | GET | Docker container info |
| `/api/resources/alerts` | GET | Resource alerts |

## 🛠️ MCP Operations

### Create MCP Node

```bash
curl -X POST "http://localhost:8080/node/create?image_name=agentdock-node:explore"
```

### Call MCP Tool

```bash
curl -X POST "http://localhost:8080/container/{container_id}/mcp" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "tool_name",
      "arguments": {}
    },
    "id": 1
  }'
```

## 🔧 Development

### Local Development

```bash
# Install dependencies
pip install -r master/requirements.txt

# Start development server
cd master
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

### Build Images

```bash
# Build all images
docker compose build

# Rebuild without cache
docker compose build --no-cache
```

### Common Commands

```bash
# Start services
docker compose up -d

# Stop services
docker compose down

# View logs
docker compose logs -f agentdock-manager

# Check status
docker compose ps

# Resource stats
docker stats
```

## 📄 License

The code in this repository is released under the [Apache-2.0](../../LICENSE) license.
