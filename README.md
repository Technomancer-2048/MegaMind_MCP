# MegaMind Context Database System

An intelligent MCP server designed to eliminate AI context exhaustion through semantic chunking and precise database retrieval.

## Overview

The MegaMind Context Database System solves the critical problem of AI context waste in development workflows. Current markdown-based systems consume 14,600+ tokens for simple tasks, making high-capability models like Opus 4 practically unusable. This system achieves **70-80% context reduction** through intelligent semantic chunking and database-driven retrieval.

## Key Features

- **Semantic Chunking**: Break documentation into 20-150 line coherent chunks
- **Intelligent Retrieval**: AI-driven context assembly with relevance scoring
- **Bidirectional Flow**: AI contributions enhance the knowledge base through review cycles
- **Model Optimization**: Tailored context delivery for Sonnet vs Opus models
- **Session Management**: Lightweight state restoration and continuity
- **Analytics Dashboard**: Usage patterns and optimization insights

## Architecture

### Core Components
- **MySQL Database**: Metadata-rich storage with cross-references and usage tracking
- **MCP Server**: Standalone interface for direct AI interaction
- **Embedding Engine**: Sentence transformers for semantic similarity
- **Analytics Dashboard**: Flask-based monitoring and visualization
- **Review Interface**: Manual approval system for knowledge updates

### Technology Stack
- **Database**: MySQL 8.0+ with full-text search and JSON support
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Caching**: Redis for performance optimization
- **Authentication**: JWT with role-based access control
- **Monitoring**: Comprehensive health checks and alerting

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.8+
- MySQL 8.0+
- Redis (for caching)

### Installation

1. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Start the database system**
   ```bash
   ./scripts/start_database.sh
   ```

3. **Ingest documentation (optional)**
   ```bash
   cd tools
   pip install -r requirements.txt
   python markdown_ingester.py /path/to/docs --password your_db_password
   ```

4. **Start the MCP server**
   ```bash
   ./scripts/start_mcp_server.sh
   ```

5. **Validate installation**
   ```bash
   ./scripts/run_tests.sh
   ```

## Configuration

### Environment Variables

```bash
# Database Configuration
MEGAMIND_DB_HOST=localhost
MEGAMIND_DB_PORT=3309
MEGAMIND_DB_NAME=megamind_database
MEGAMIND_DB_USER=megamind_user
MEGAMIND_DB_PASSWORD=secure_password

# MCP Server Configuration
MEGAMIND_MCP_SERVER_PORT=8002
MEGAMIND_MCP_AUTH_SECRET=jwt_secret_key
MEGAMIND_MCP_RATE_LIMIT_REQUESTS=100

# Performance Configuration
REDIS_URL=redis://localhost:6379/3
EMBEDDING_CACHE_SIZE=1000
CONNECTION_POOL_SIZE=10
```

### Docker Configuration

The system uses Docker Compose for easy deployment:

```yaml
# docker-compose.megamind-db.yml
services:
  megamind-db-mysql:
    image: mysql:8.0
    ports:
      - "3309:3306"
  
  megamind-mcp-server:
    build: ./mcp_server
    ports:
      - "8002:8002"
    depends_on:
      - megamind-db-mysql
```

## MCP Functions

### Core Retrieval Functions
- `mcp__megamind_db__search_chunks(query, limit=10, model_type="sonnet")` - Semantic search with model optimization
- `mcp__megamind_db__get_chunk(chunk_id, include_relationships=true)` - Retrieve specific chunk with metadata
- `mcp__megamind_db__get_related_chunks(chunk_id, max_depth=2)` - Traverse relationship graph

### Knowledge Management Functions
- `mcp__megamind_db__update_chunk(chunk_id, new_content, session_id)` - Buffer chunk modifications
- `mcp__megamind_db__create_chunk(content, source_document, section_path, session_id)` - Create new chunks
- `mcp__megamind_db__commit_session_changes(session_id, approved_changes)` - Commit approved changes

### Session Management Functions
- `mcp__megamind_db__get_session_primer(last_session_data)` - Generate context for session continuity
- `mcp__megamind_db__track_access(chunk_id, query_context)` - Update access statistics

## Development Phases

### Phase 1: Core Infrastructure (Weeks 1-2) ✅ COMPLETED
- ✅ Database schema design with optimized indexing
- ✅ Markdown ingestion tool for existing documentation
- ✅ Basic MCP server with core retrieval functions
- ✅ Docker configuration and deployment scripts
- ✅ Comprehensive validation test suite

### Phase 2: Intelligence Layer (Weeks 3-4)
- 🔄 Semantic analysis engine with embedding generation
- 🔄 Context analytics dashboard for usage monitoring
- 🔄 Enhanced MCP functions with relationship traversal

### Phase 3: Bidirectional Flow (Weeks 5-6)
- ⏳ Knowledge update functions with session buffering
- ⏳ Manual review interface for change approval
- ⏳ Change management and rollback capabilities

### Phase 4: Advanced Optimization (Weeks 7-8)
- ⏳ Model-specific optimization (Sonnet vs Opus)
- ⏳ Automated curation system
- ⏳ Comprehensive system health monitoring

## Performance Targets

- **Context Reduction**: 70-80% reduction in token consumption
- **Response Time**: < 200ms for 95th percentile queries
- **Concurrent Users**: Support 50+ simultaneous sessions
- **Accuracy**: >80% semantic relationship discovery
- **Availability**: 99.9% uptime with automated failover

## Testing

### Test Categories
- **Unit Tests**: 80+ tests covering MCP functions and database operations
- **Integration Tests**: 20+ tests for end-to-end workflows
- **Performance Tests**: 10+ benchmarks for response time and concurrency

### Running Tests
```bash
# Unit tests
python -m pytest tests/unit/

# Integration tests
python -m pytest tests/integration/

# Performance benchmarks
python -m pytest tests/performance/ --benchmark
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

**Status**: Phase 1 implementation ready with comprehensive technical specifications.