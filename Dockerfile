# MegaMind Context Database - Production Container
# Includes complete semantic search system with realm-aware architecture
FROM python:3.11-slim

LABEL maintainer="MegaMind Context Database Team"
LABEL version="4.0.0"
LABEL description="Complete realm-aware semantic search system with performance optimization"

# Set working directory
WORKDIR /app

# Set environment variables for Python and container optimization
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for MySQL, build tools, and optimization
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    default-libmysqlclient-dev \
    build-essential \
    procps \
    curl \
    vim \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r megamind && useradd -r -g megamind -s /bin/bash megamind

# Copy requirements first for better layer caching
COPY mcp_server/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Create directory structure
RUN mkdir -p /app/services /app/database /app/scripts /app/tools /app/tests

# Copy MCP server core components - copy all Python files and core files
COPY mcp_server/*.py ./
COPY mcp_server/init_schema.sql ./
COPY mcp_server/entrypoint.sh ./

# Copy Phase 4 performance optimization services
COPY mcp_server/services/ ./services/

# Copy database schemas and configurations
COPY database/ ./database/

# Copy tools and utilities
COPY tools/ ./tools/

# Copy validation and testing scripts
COPY scripts/validate_realm_semantic_search.py ./scripts/
COPY tests/benchmark_realm_semantic_search.py ./tests/

# Copy configuration files
COPY mcp.json ./
COPY CLAUDE.md ./

# Make scripts executable
RUN chmod +x entrypoint.sh \
    && chmod +x scripts/*.py 2>/dev/null || true \
    && chmod +x tests/*.py 2>/dev/null || true

# Create data directories and set permissions
RUN mkdir -p /app/data /app/logs /app/cache /app/.cache \
    && chown -R megamind:megamind /app \
    && chmod -R 755 /app

# Set default environment variables for semantic search
ENV EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV EMBEDDING_DEVICE=cpu
ENV EMBEDDING_BATCH_SIZE=50
ENV EMBEDDING_CACHE_SIZE=1000
ENV EMBEDDING_CACHE_TTL=3600
ENV SEMANTIC_SEARCH_THRESHOLD=0.7
ENV REALM_PRIORITY_PROJECT=1.2
ENV REALM_PRIORITY_GLOBAL=1.0
ENV CROSS_REALM_SEARCH_ENABLED=true

# Set unified cache directories for HuggingFace and sentence-transformers
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface
ENV HF_HUB_CACHE=/app/.cache/huggingface

# MCP Server configuration
ENV MEGAMIND_MCP_SERVER_PORT=8002
ENV CONNECTION_POOL_SIZE=10

# Database configuration (will be overridden by docker-compose)
ENV MEGAMIND_DB_HOST=localhost
ENV MEGAMIND_DB_PORT=3306
ENV MEGAMIND_DB_NAME=megamind_database
ENV MEGAMIND_DB_USER=megamind_user
ENV MEGAMIND_DB_PASSWORD=megamind_secure_pass

# Realm configuration defaults
ENV MEGAMIND_PROJECT_REALM=PROJ_DEFAULT
ENV MEGAMIND_DEFAULT_TARGET=PROJECT

# Performance optimization settings
ENV ASYNC_EMBEDDING_ENABLED=true
ENV ASYNC_MAX_WORKERS=3
ENV ASYNC_BATCH_SIZE=10

# Expose MCP server port
EXPOSE 8002

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8002/health', timeout=5)" || exit 1

# Switch to non-root user
USER megamind

# Set volume for persistent data
VOLUME ["/app/data", "/app/logs", "/app/cache"]

# Run the MCP server with initialization
CMD ["./entrypoint.sh"]