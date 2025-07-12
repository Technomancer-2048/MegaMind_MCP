#!/bin/bash
# MegaMind Context Database - Clean Container Build Script
# Builds a clean production container with all Phase 4 optimizations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="megamind-context-db"
VERSION="4.0.0"
REGISTRY_PREFIX=""  # Set to your registry prefix if needed

echo -e "${BLUE}🚀 MegaMind Context Database - Clean Container Build${NC}"
echo -e "${BLUE}====================================================${NC}"

# Function to print status
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

# Check prerequisites
echo -e "${BLUE}🔍 Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi
print_status "Docker found"

if ! command -v docker-compose &> /dev/null; then
    print_warning "docker-compose not found, trying docker compose..."
    if ! docker compose version &> /dev/null; then
        print_error "Neither docker-compose nor 'docker compose' found"
        exit 1
    fi
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi
print_status "Docker Compose found"

# Check if we're in the right directory
if [ ! -f "Dockerfile" ] || [ ! -f "docker-compose.prod.yml" ]; then
    print_error "Must be run from the MegaMind_MCP root directory"
    exit 1
fi
print_status "In correct directory"

# Create data directories
echo -e "${BLUE}📁 Creating data directories...${NC}"
mkdir -p data/mysql data/redis data/mcp logs cache
print_status "Data directories created"

# Clean up any previous builds
echo -e "${BLUE}🧹 Cleaning up previous builds...${NC}"
docker image prune -f --filter="label=version=4.0.0" 2>/dev/null || true
print_status "Previous builds cleaned"

# Build the main application image
echo -e "${BLUE}🔨 Building MegaMind Context Database image...${NC}"
docker build \
    --tag ${REGISTRY_PREFIX}${IMAGE_NAME}:${VERSION} \
    --tag ${REGISTRY_PREFIX}${IMAGE_NAME}:latest \
    --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
    --build-arg VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown") \
    --progress=plain \
    .

if [ $? -eq 0 ]; then
    print_status "Main image built successfully"
else
    print_error "Failed to build main image"
    exit 1
fi

# Build dashboard image if Dockerfile exists
if [ -f "dashboard/Dockerfile" ]; then
    echo -e "${BLUE}🔨 Building dashboard image...${NC}"
    docker build \
        --tag ${REGISTRY_PREFIX}megamind-dashboard:1.0.0 \
        --tag ${REGISTRY_PREFIX}megamind-dashboard:latest \
        ./dashboard
    
    if [ $? -eq 0 ]; then
        print_status "Dashboard image built successfully"
    else
        print_warning "Dashboard image build failed (optional)"
    fi
else
    print_warning "Dashboard Dockerfile not found, skipping dashboard build"
fi

# Validate the build
echo -e "${BLUE}✅ Validating container build...${NC}"

# Check if image exists
if docker images ${REGISTRY_PREFIX}${IMAGE_NAME}:${VERSION} --format "table {{.Repository}}:{{.Tag}}" | grep -q ${VERSION}; then
    print_status "Image ${IMAGE_NAME}:${VERSION} created successfully"
else
    print_error "Image validation failed"
    exit 1
fi

# Get image size
IMAGE_SIZE=$(docker images ${REGISTRY_PREFIX}${IMAGE_NAME}:${VERSION} --format "{{.Size}}")
print_status "Image size: ${IMAGE_SIZE}"

# Quick container test
echo -e "${BLUE}🧪 Running quick container test...${NC}"
CONTAINER_ID=$(docker run -d --rm \
    -e MEGAMIND_DB_HOST=test \
    -e MEGAMIND_DB_PASSWORD=test \
    ${REGISTRY_PREFIX}${IMAGE_NAME}:${VERSION} \
    python -c "print('Container test successful'); import time; time.sleep(5)")

if [ $? -eq 0 ]; then
    print_status "Container starts successfully"
    docker stop ${CONTAINER_ID} &>/dev/null || true
else
    print_error "Container test failed"
    exit 1
fi

# Display usage instructions
echo -e "${BLUE}📋 Build Complete!${NC}"
echo -e "${BLUE}=================${NC}"
echo ""
echo -e "${GREEN}Image built successfully:${NC} ${REGISTRY_PREFIX}${IMAGE_NAME}:${VERSION}"
echo -e "${GREEN}Image size:${NC} ${IMAGE_SIZE}"
echo ""
echo -e "${YELLOW}Usage Instructions:${NC}"
echo ""
echo -e "${BLUE}1. Development (quick start):${NC}"
echo "   docker-compose up -d"
echo ""
echo -e "${BLUE}2. Production deployment:${NC}"
echo "   cp .env.production .env"
echo "   # Edit .env with your configuration"
echo "   ${COMPOSE_CMD} -f docker-compose.prod.yml up -d"
echo ""
echo -e "${BLUE}3. With dashboard:${NC}"
echo "   ${COMPOSE_CMD} -f docker-compose.prod.yml --profile dashboard up -d"
echo ""
echo -e "${BLUE}4. Run validation:${NC}"
echo "   docker run --rm ${REGISTRY_PREFIX}${IMAGE_NAME}:${VERSION} python scripts/validate_realm_semantic_search.py --help"
echo ""
echo -e "${BLUE}5. Run benchmarks:${NC}"
echo "   docker run --rm ${REGISTRY_PREFIX}${IMAGE_NAME}:${VERSION} python tests/benchmark_realm_semantic_search.py --help"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "   - Copy .env.production to .env and customize"
echo "   - Default ports: MySQL(3309), Redis(6379), MCP(8002), Dashboard(8080)"
echo "   - Data persisted in ./data/ directory"
echo "   - Logs available in ./logs/ directory"
echo ""
echo -e "${GREEN}🎉 Clean container build completed successfully!${NC}"

# Optional: Show image details
if [ "$1" = "--verbose" ] || [ "$1" = "-v" ]; then
    echo ""
    echo -e "${BLUE}📊 Image Details:${NC}"
    docker inspect ${REGISTRY_PREFIX}${IMAGE_NAME}:${VERSION} --format='
🏷️  Image: {{.RepoTags}}
📅 Created: {{.Created}}
🏗️  Architecture: {{.Architecture}}
💾 Size: {{.Size}} bytes
🔧 Entrypoint: {{.Config.Entrypoint}}
📝 CMD: {{.Config.Cmd}}
🌍 Environment Variables:
{{- range .Config.Env}}
   {{.}}
{{- end}}
🚪 Exposed Ports:
{{- range $port, $config := .Config.ExposedPorts}}
   {{$port}}
{{- end}}'
fi