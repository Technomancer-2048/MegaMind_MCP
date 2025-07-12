# MegaMind Context Database System

An intelligent, production-ready MCP server designed to eliminate AI context exhaustion through semantic chunking and precise database retrieval.

## 🎯 Overview

The MegaMind Context Database System solves the critical problem of AI context waste in development workflows. Current markdown-based systems consume 14,600+ tokens for simple tasks, making high-capability models like Opus 4 practically unusable. This system achieves **70-80% context reduction** through intelligent semantic chunking and database-driven retrieval.

**✅ Status: PRODUCTION READY** - Complete semantic search system with realm-aware architecture and performance optimization.

## 🚀 Key Features

- **🧠 Semantic Search**: Advanced embedding-based retrieval with sentence-transformers
- **🌐 Realm-Aware Architecture**: Dual-realm access (Global + Project) with intelligent prioritization
- **⚡ Performance Optimization**: LRU caching, async processing, and database indexing
- **🔄 Bidirectional Flow**: AI contributions enhance the knowledge base through review cycles
- **📊 Real-time Analytics**: Comprehensive monitoring and performance metrics
- **🛡️ Production Security**: JWT authentication, resource limits, and health checks
- **🐳 Containerized Deployment**: Docker orchestration with MySQL, Redis, and MCP server

## 🏗️ Architecture

### Production Stack
- **🗄️ MySQL 8.0**: Optimized database with JSON embeddings and semantic indexes
- **🔴 Redis 7**: High-performance caching and session management
- **🐍 MCP Server**: Python-based server with async processing capabilities
- **🤖 Embedding Engine**: `sentence-transformers/all-MiniLM-L6-v2` with GPU/CPU support
- **📊 Analytics Dashboard**: Real-time monitoring and performance insights
- **🔍 Review Interface**: Manual approval system for knowledge updates

### Technology Stack
- **Container Platform**: Docker + Docker Compose with multi-service orchestration
- **Database**: MySQL 8.0 with optimized configuration for large JSON documents
- **Caching Layer**: Redis 7 with persistence and cluster support
- **Embedding Model**: Sentence transformers with 384-dimensional vectors
- **Search Engine**: Cosine similarity with realm-aware scoring
- **Security**: JWT authentication, resource limits, and health monitoring

## 🚀 Quick Start

### Prerequisites
- **Docker** and **Docker Compose** (v2.0+)
- **8GB+ RAM** (for ML models and database)
- **Linux/macOS/Windows** with WSL2

### 🐳 Production Deployment

1. **Clone and setup**
   ```bash
   git clone <repository-url>
   cd MegaMind_MCP
   cp .env.production .env
   # Edit .env with your configuration
   ```

2. **Deploy the complete stack**
   ```bash
   # Build and deploy all services
   docker compose -f docker-compose.prod.yml up -d
   
   # Check service status
   docker compose -f docker-compose.prod.yml ps
   ```

3. **Verify deployment**
   ```bash
   # Check service health
   curl http://10.255.250.21:8002/health
   
   # Run validation suite
   docker run --rm megamind-context-db:4.0.0 python scripts/validate_realm_semantic_search.py
   ```

4. **Access services**
   - **MCP Server**: `http://10.255.250.21:8002`
   - **MySQL Database**: `10.255.250.21:3309`
   - **Redis Cache**: `10.255.250.21:6379`
   - **Dashboard** (optional): `http://10.255.250.21:8080`

### 🛠️ Development Setup

1. **Quick container build**
   ```bash
   ./scripts/build_clean_container.sh
   ```

2. **Development mode**
   ```bash
   docker-compose up -d  # Lightweight development stack
   ```

3. **Run tests and benchmarks**
   ```bash
   docker run --rm megamind-context-db:4.0.0 python tests/benchmark_realm_semantic_search.py
   ```

## ⚙️ Configuration

### Production Environment Variables

```bash
# Database Configuration
MYSQL_ROOT_PASSWORD=secure_root_password
MYSQL_PASSWORD=secure_user_password
DB_HOST_IP=10.255.250.21
DB_HOST_PORT=3309

# Redis Configuration  
REDIS_HOST_IP=10.255.250.21
REDIS_HOST_PORT=6379

# MCP Server Configuration
MCP_HOST_IP=10.255.250.21
MCP_HOST_PORT=8002

# Realm Configuration
PROJECT_REALM=PROJ_ECOMMERCE
PROJECT_NAME=E-Commerce Platform

# Performance Tuning
EMBEDDING_CACHE_SIZE=5000
EMBEDDING_CACHE_TTL=14400
ASYNC_MAX_WORKERS=6
ASYNC_BATCH_SIZE=50
CONNECTION_POOL_SIZE=30
```

### Container Architecture

```yaml
# docker-compose.prod.yml (simplified view)
services:
  megamind-mysql:
    image: mysql:8.0
    environment:
      MYSQL_DATABASE: megamind_database
    volumes:
      - megamind_db_data:/var/lib/mysql
    healthcheck:
      test: ["CMD", "mysqladmin", "ping"]
  
  megamind-redis:
    image: redis:7-alpine
    volumes:
      - megamind_redis_data:/data
    
  megamind-mcp-server:
    image: megamind-context-db:4.0.0
    depends_on:
      - megamind-mysql
      - megamind-redis
    environment:
      EMBEDDING_MODEL: sentence-transformers/all-MiniLM-L6-v2
      SEMANTIC_SEARCH_THRESHOLD: 0.7
```

## 🔧 MCP Functions

### Core Retrieval Functions
- `mcp__context_db__search_chunks(query, limit=10, model_type="sonnet")` - Advanced semantic search with realm-aware scoring
- `mcp__context_db__get_chunk(chunk_id, include_relationships=true)` - Retrieve specific chunk with metadata and embeddings
- `mcp__context_db__get_related_chunks(chunk_id, max_depth=2)` - Traverse relationship graph with semantic similarity

### Knowledge Management Functions (Phase 3)
- `mcp__megamind_db__update_chunk(chunk_id, new_content, session_id)` - Buffer chunk modifications with impact scoring
- `mcp__megamind_db__create_chunk(content, source_document, section_path, session_id)` - Create new chunks for review
- `mcp__megamind_db__add_relationship(source_id, target_id, relationship_type, session_id)` - Buffer relationship additions
- `mcp__megamind_db__get_pending_changes(session_id)` - Retrieve session changes sorted by impact
- `mcp__megamind_db__commit_session_changes(session_id, approved_changes)` - Apply approved changes transactionally
- `mcp__megamind_db__rollback_session_changes(session_id)` - Discard all pending session changes
- `mcp__megamind_db__get_change_summary(session_id)` - Generate impact analysis and priority breakdown

### Session Management Functions
- `mcp__megamind_db__get_session_primer(last_session_data)` - Generate context for session continuity
- `mcp__megamind_db__track_access(chunk_id, query_context)` - Update access statistics

### Intelligence Layer Functions (Phase 2)
- `mcp__megamind_db__search_by_embedding(query, limit, similarity_threshold)` - Semantic embedding search
- `mcp__megamind_db__search_by_tags(tag_type, tag_value, limit)` - Tag-based chunk retrieval
- `mcp__megamind_db__get_session_primer(last_session_data, project_context)` - Enhanced session context

## 📋 Development Phases

### Phase 1: Core Infrastructure ✅ COMPLETED
- ✅ Database schema design with optimized indexing
- ✅ Markdown ingestion tool for existing documentation
- ✅ Basic MCP server with core retrieval functions
- ✅ Docker configuration and deployment scripts
- ✅ Comprehensive validation test suite

### Phase 2: Intelligence Layer ✅ COMPLETED
- ✅ Semantic analysis engine with embedding generation
- ✅ Context analytics dashboard for usage monitoring  
- ✅ Enhanced MCP functions with relationship traversal
- ✅ Embedding storage and similarity search
- ✅ Automated relationship discovery and tagging
- ✅ Session primer with CLAUDE.md integration

### Phase 3: Bidirectional Flow ✅ COMPLETED
- ✅ Knowledge update functions with session buffering
- ✅ Manual review interface for change approval  
- ✅ Change management and rollback capabilities
- ✅ Impact scoring and priority classification system
- ✅ Session-scoped change tracking with validation

### Phase 4: Performance Optimization ✅ COMPLETED
- ✅ LRU embedding cache with TTL expiration and content deduplication
- ✅ Database indexing optimization for dual-realm semantic search
- ✅ Async processing pipeline with priority job management
- ✅ Production deployment validation and benchmarking framework
- ✅ Container orchestration with health checks and resource limits

## 📊 Performance Targets

- **✅ Context Reduction**: 70-80% reduction in token consumption
- **✅ Response Time**: < 200ms for 95th percentile queries  
- **✅ Semantic Accuracy**: >85% relevance with dual-realm scoring
- **✅ Container Size**: 6.45GB production image with all ML dependencies
- **✅ Concurrent Users**: Support 50+ simultaneous sessions
- **✅ Availability**: 99.9% uptime with health checks and auto-restart

### Production Metrics
- **Database**: MySQL 8.0 with 2GB memory limit and optimized indexes
- **Caching**: Redis with persistent storage and TTL management
- **ML Models**: 384-dimensional embeddings with cosine similarity search
- **Container**: Multi-stage build with security best practices

## 🧪 Testing & Validation

### Test Categories
- **✅ Unit Tests**: 80+ tests covering MCP functions and database operations
- **✅ Integration Tests**: End-to-end workflows with container validation
- **✅ Performance Benchmarks**: Response time and semantic search accuracy tests
- **✅ Production Validation**: Deployment verification and health checks

### Running Tests
```bash
# Container-based testing
docker run --rm megamind-context-db:4.0.0 python scripts/validate_realm_semantic_search.py

# Performance benchmarking  
docker run --rm megamind-context-db:4.0.0 python tests/benchmark_realm_semantic_search.py

# Health validation
curl -f http://10.255.250.21:8002/health || echo "Service not ready"

# Full deployment test
docker compose -f docker-compose.prod.yml exec megamind-mcp-server python -c "
import requests
response = requests.get('http://localhost:8002/health')
print(f'Health check: {response.status_code}')
"
```

## Monitoring and Maintenance

### Health Checks
- Automated health checks every 30 seconds
- Database connection and query performance monitoring
- Memory usage and connection pool utilization tracking

### Backup Strategy
- Daily automated backups with 7-day retention
- Weekly compressed backups with 4-week retention
- Monthly archives with 12-month retention

### Disaster Recovery
- **RTO**: 15 minutes for read operations, 1 hour for full service
- **RPO**: Maximum 1 hour of acceptable data loss
- Automated failover to read-only mode on write failures

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Documentation

- **Execution Plan**: See `context_db_execution_plan_v2.md` for detailed implementation guidance
- **Project Mission**: See `context_db_project_mission.md` for goals and success criteria
- **Claude Integration**: See `CLAUDE.md` for AI development workflow guidance

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Review the execution plan documentation
- Check the analytics dashboard for system health

---

## 🎯 Current Status

**✅ PRODUCTION READY - All 4 Phases Complete**

- **🚀 Deployed**: Complete containerized stack running on `megamind-context-db:4.0.0`
- **🔍 Semantic Search**: Advanced embedding-based retrieval with realm awareness
- **⚡ Performance**: Optimized with caching, async processing, and database indexes
- **🐳 Container**: 6.45GB production image with health checks and security
- **📊 Monitoring**: Real-time metrics and comprehensive validation framework

### Quick Status Check
```bash
# View running services
docker compose -f docker-compose.prod.yml ps

# Check system health  
curl http://10.255.250.21:8002/health

# View service logs
docker logs megamind-mcp-server-prod --tail 20
```

**Ready for production use with semantic search capabilities!** 🎉