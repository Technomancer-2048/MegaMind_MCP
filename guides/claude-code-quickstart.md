# MegaMind MCP Quick Start Guide for Claude Code

This guide helps you set up and configure the MegaMind Context Database MCP server for use with Claude Code (claude.ai/code).

## 🎯 What You'll Get

After following this guide, you'll have:
- **20 MCP functions** for intelligent context management
- **Semantic search** capabilities across your knowledge base
- **Knowledge promotion system** for cross-project knowledge sharing
- **70-80% reduction** in context token consumption
- **Sub-second retrieval** for development workflows

## 📋 Prerequisites

### System Requirements
- **Docker** and **Docker Compose** (v2.0+)
- **8GB+ RAM** (for ML models and database)
- **10GB+ free disk space** (for container images and data)
- **Linux, macOS, or Windows** with WSL2

### Claude Code Setup
- Active Claude Code subscription
- Basic familiarity with MCP (Model Context Protocol)
- Text editor for configuration files

## 🚀 Step 1: Clone and Setup

### 1.1 Clone the Repository
```bash
git clone https://github.com/Technomancer-2048/MegaMind_MCP.git
cd MegaMind_MCP
```

### 1.2 Copy Configuration Template
```bash
# Copy the environment template
cp .env.template .env

# Copy the MCP configuration template  
cp Guides/mcp.json ~/.config/claude-code/mcp.json
```

### 1.3 Configure Environment Variables
Edit the `.env` file with your settings:

```bash
# Database Configuration
MEGAMIND_DB_HOST=localhost
MEGAMIND_DB_PORT=3306
MEGAMIND_DB_NAME=megamind_database
MEGAMIND_DB_USER=megamind_user
MEGAMIND_DB_PASSWORD=your_secure_password_here

# MCP Server Configuration
MEGAMIND_PROJECT_REALM=YOUR_PROJECT_NAME
MEGAMIND_PROJECT_NAME="Your Project Display Name"
MEGAMIND_DEFAULT_TARGET=PROJECT

# Optional: Performance Tuning
EMBEDDING_CACHE_SIZE=5000
EMBEDDING_CACHE_TTL=14400
```

**Important**: Replace `your_secure_password_here` with a strong password.

## 🐳 Step 2: Deploy the MCP Server

### 2.1 Start the Complete Stack
```bash
# Build and start all services
docker compose up -d

# Verify services are running
docker compose ps
```

Expected output:
```
NAME                     IMAGE                                   STATUS
megamind-mysql           mysql:8.0                              Up (healthy)
megamind-redis           redis:7-alpine                         Up (healthy)  
megamind-mcp-server-http megamind_mcp-megamind-mcp-server-http  Up (healthy)
```

### 2.2 Verify MCP Server Health
```bash
# Check server health
curl http://localhost:8080/mcp/health

# Expected response: {"status": "healthy", "timestamp": "..."}
```

### 2.3 Test Database Connection
```bash
# Run the comprehensive test suite
./test_all_mcp_functions.sh

# Should show: "✅ All MCP functions working correctly"
```

## ⚙️ Step 3: Configure Claude Code

### 3.1 Locate Claude Code Configuration
The MCP configuration file should be at:
- **macOS**: `~/.config/claude-code/mcp.json`
- **Linux**: `~/.config/claude-code/mcp.json`  
- **Windows**: `%APPDATA%\claude-code\mcp.json`

### 3.2 Configure MCP Connection
Edit your `mcp.json` file:

```json
{
  "servers": {
    "megamind-context-db": {
      "command": "docker",
      "args": [
        "exec", "-i", "megamind-mcp-server-http",
        "python", "-m", "mcp_server.megamind_database_server"
      ],
      "env": {
        "MEGAMIND_PROJECT_REALM": "YOUR_PROJECT_NAME",
        "MEGAMIND_PROJECT_NAME": "Your Project Display Name",
        "MEGAMIND_DEFAULT_TARGET": "PROJECT"
      }
    }
  }
}
```

### 3.3 Alternative: HTTP Transport (Advanced)
For HTTP-based connection (requires additional setup):

```json
{
  "servers": {
    "megamind-context-db": {
      "command": "curl",
      "args": [
        "-X", "POST",
        "-H", "Content-Type: application/json",
        "--data-raw", "{}",
        "http://localhost:8080/mcp/jsonrpc"
      ]
    }
  }
}
```

## 🧪 Step 4: Test the Integration

### 4.1 Restart Claude Code
Close and restart Claude Code to load the new MCP configuration.

### 4.2 Verify MCP Functions
In a Claude Code session, test the connection:

1. **Basic Search Test**:
   ```
   Search for chunks about "database configuration" using MegaMind MCP
   ```

2. **Function Availability Check**:
   ```
   Show me all available MegaMind MCP functions
   ```

3. **Semantic Search Test**:
   ```
   Use semantic search to find content similar to "error handling patterns"
   ```

### 4.3 Expected Results
You should see Claude Code successfully:
- Connect to the MegaMind MCP server
- List 20 available MCP functions
- Perform semantic searches across your knowledge base
- Retrieve relevant context chunks with metadata

## 📚 Step 5: Load Your Knowledge Base

### 5.1 Prepare Your Documentation
Organize your project documentation:
```
docs/
├── api/
│   ├── authentication.md
│   └── endpoints.md
├── deployment/
│   ├── docker.md
│   └── kubernetes.md
└── development/
    ├── setup.md
    └── testing.md
```

### 5.2 Ingest Documentation (Manual Method)
Use the content management functions through Claude Code:

```
Create a new chunk with this content:
- Source: "docs/api/authentication.md"  
- Section: "JWT Token Validation"
- Content: [paste your documentation content]
```

### 5.3 Batch Ingestion (Advanced)
For large documentation sets, use the batch ingestion script:

```bash
# Copy documentation to container
docker cp docs/ megamind-mcp-server-http:/tmp/docs/

# Run batch ingestion
docker exec megamind-mcp-server-http python scripts/ingest_documentation.py /tmp/docs/
```

## 🔧 Step 6: Using MCP Functions

### 6.1 Search Functions
```
# Basic search
Search for "error handling" using MegaMind

# Semantic search  
Find content semantically similar to "authentication patterns"

# Get related chunks
Show me chunks related to chunk ID "chunk_12345"
```

### 6.2 Knowledge Promotion
```
# Create promotion request
Create a promotion request for chunk "chunk_67890" to GLOBAL realm with justification "Broadly applicable error handling pattern"

# Review promotion queue
Show me the current promotion queue summary

# Approve promotion
Approve promotion request "promo_123" with reason "High-value pattern confirmed"
```

### 6.3 Session Management
```
# Get session primer
Generate a session primer based on my recent work

# Track access
Track access to chunk "chunk_45678" with context "debugging API issues"
```

## 🎛️ Customization Options

### 6.1 Realm Configuration
Customize your realm setup in `.env`:
```bash
# Multi-project setup
MEGAMIND_PROJECT_REALM=ECOMMERCE_API
MEGAMIND_PROJECT_NAME="E-Commerce API Platform"

# or
MEGAMIND_PROJECT_REALM=MOBILE_APP  
MEGAMIND_PROJECT_NAME="Mobile Application"
```

### 6.2 Performance Tuning
Adjust performance settings:
```bash
# For large projects
EMBEDDING_CACHE_SIZE=10000
EMBEDDING_CACHE_TTL=21600

# For smaller projects  
EMBEDDING_CACHE_SIZE=2000
EMBEDDING_CACHE_TTL=7200
```

### 6.3 Search Behavior
Configure search preferences:
```bash
# Stricter semantic matching
SEMANTIC_SEARCH_THRESHOLD=0.8

# More lenient matching
SEMANTIC_SEARCH_THRESHOLD=0.6
```

## 🔍 Troubleshooting

### Common Issues

**1. "MCP server not responding"**
```bash
# Check container status
docker compose ps

# Check logs
docker logs megamind-mcp-server-http --tail 20

# Restart if needed
docker compose restart megamind-mcp-server-http
```

**2. "Database connection failed"**
```bash
# Verify database is running
docker exec megamind-mysql mysqladmin -u megamind_user -p ping

# Check database logs
docker logs megamind-mysql --tail 20
```

**3. "Embedding service not ready"**
```bash
# Wait for embedding model download (first startup takes 2-3 minutes)
docker logs megamind-mcp-server-http | grep -i "embedding"

# Check available memory
docker stats megamind-mcp-server-http
```

**4. "Functions returning empty results"**
```bash
# Check if data exists
docker exec megamind-mysql mysql -u megamind_user -p megamind_database -e "SELECT COUNT(*) FROM megamind_chunks;"

# If empty, ingest some test data
echo "Test content" | docker exec -i megamind-mcp-server-http python scripts/quick_ingest.py
```

### Performance Optimization

**For Better Response Times**:
```bash
# Increase cache size
EMBEDDING_CACHE_SIZE=10000

# Add more worker threads
ASYNC_MAX_WORKERS=8

# Increase connection pool
CONNECTION_POOL_SIZE=40
```

**For Memory Constrained Systems**:
```bash
# Reduce cache size
EMBEDDING_CACHE_SIZE=1000

# Fewer workers
ASYNC_MAX_WORKERS=2

# Smaller connection pool
CONNECTION_POOL_SIZE=10
```

## 📊 Monitoring Usage

### Health Checks
```bash
# Quick health check
curl http://localhost:8080/mcp/health

# Detailed status
curl http://localhost:8080/mcp/status

# Realm information
curl http://localhost:8080/mcp/realms
```

### Usage Analytics
Monitor your usage through Claude Code:
```
Show me hot contexts and frequently accessed chunks
Get promotion queue summary to see knowledge sharing activity
Track access patterns for optimization insights
```

## 🚀 Next Steps

### Advanced Features
1. **Multi-Project Setup**: Configure multiple realm contexts
2. **Batch Operations**: Use batch embedding generation for large datasets  
3. **Custom Analytics**: Set up monitoring dashboards
4. **API Integration**: Use HTTP endpoints for programmatic access

### Best Practices
1. **Regular Promotion Reviews**: Check promotion queue weekly
2. **Knowledge Curation**: Promote valuable patterns to GLOBAL realm
3. **Performance Monitoring**: Track response times and cache hit rates
4. **Content Organization**: Use meaningful source document names and section paths

### Community Resources
- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Full technical documentation in `CLAUDE.md`
- **Examples**: Sample configurations in `Guides/` folder

## 📞 Support

If you encounter issues:

1. **Check the logs**: `docker logs megamind-mcp-server-http`
2. **Verify configuration**: Review `.env` and `mcp.json` settings
3. **Test connectivity**: Run health checks and function tests
4. **GitHub Issues**: Open an issue with logs and configuration details

---

**🎉 Congratulations!** You now have a fully functional MegaMind MCP server integrated with Claude Code, providing intelligent context management and semantic search capabilities for your development workflows.

For advanced configuration and technical details, see the complete documentation in `CLAUDE.md`.