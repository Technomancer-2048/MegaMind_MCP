#!/bin/bash
# MegaMind MCP Server - Phase 3 Container Deployment Script
# Comprehensive deployment automation for HTTP transport

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEFAULT_ENV_FILE="$PROJECT_DIR/.env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Usage function
usage() {
    cat << EOF
MegaMind MCP Server - Phase 3 Container Deployment

Usage: $0 [OPTIONS] [COMMAND]

Commands:
    start       Start HTTP MCP server (default)
    stop        Stop all services
    restart     Restart HTTP MCP server
    status      Show service status
    logs        Show service logs
    health      Run health check
    build       Build containers
    clean       Clean up containers and volumes
    legacy      Start legacy stdio server

Options:
    -e, --env FILE      Environment file (default: .env)
    -p, --profile NAME  Docker compose profile
    -h, --help         Show this help message

Examples:
    $0 start                    # Start HTTP server with default config
    $0 start --profile legacy   # Start both HTTP and stdio servers
    $0 health                   # Run health check
    $0 logs megamind-mcp-server-http  # Show HTTP server logs

EOF
}

# Environment setup
setup_environment() {
    local env_file="${1:-$DEFAULT_ENV_FILE}"
    
    log_info "Setting up environment..."
    
    # Check if environment file exists
    if [[ ! -f "$env_file" ]]; then
        if [[ -f "$PROJECT_DIR/.env.template" ]]; then
            log_warning "Environment file not found. Creating from template..."
            cp "$PROJECT_DIR/.env.template" "$env_file"
            log_warning "Please edit $env_file with your configuration before proceeding."
            return 1
        else
            log_error "No environment file found and no template available."
            return 1
        fi
    fi
    
    # Export environment variables
    set -a
    source "$env_file"
    set +a
    
    log_success "Environment loaded from $env_file"
    return 0
}

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        return 1
    fi
    
    if ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed or not in PATH"
        return 1
    fi
    
    # Check required files
    local required_files=(
        "$PROJECT_DIR/docker-compose.yml"
        "$PROJECT_DIR/Dockerfile.http-server"
        "$PROJECT_DIR/mcp_server/http_server.py"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "Required file not found: $file"
            return 1
        fi
    done
    
    # Check environment variables
    local required_vars=(
        "MEGAMIND_DB_PASSWORD"
        "MYSQL_ROOT_PASSWORD"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "Required environment variable not set: $var"
            return 1
        fi
    done
    
    log_success "Pre-deployment checks passed"
    return 0
}

# Build containers
build_containers() {
    log_info "Building containers..."
    
    cd "$PROJECT_DIR"
    
    # Build HTTP server container
    docker compose build megamind-mcp-server-http
    
    log_success "Containers built successfully"
}

# Start services
start_services() {
    local profile="${1:-}"
    
    log_info "Starting MegaMind MCP services..."
    
    cd "$PROJECT_DIR"
    
    if [[ -n "$profile" ]]; then
        log_info "Using profile: $profile"
        docker compose --profile "$profile" up -d
    else
        # Default: start HTTP server and dependencies
        docker compose up -d megamind-mysql megamind-redis megamind-mcp-server-http
    fi
    
    log_success "Services started"
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 10
    
    # Check service health
    check_service_health
}

# Stop services
stop_services() {
    log_info "Stopping MegaMind MCP services..."
    
    cd "$PROJECT_DIR"
    docker compose down
    
    log_success "Services stopped"
}

# Restart services
restart_services() {
    log_info "Restarting MegaMind MCP services..."
    
    stop_services
    start_services
}

# Show service status
show_status() {
    log_info "Service Status:"
    
    cd "$PROJECT_DIR"
    docker compose ps
    
    echo ""
    log_info "Container Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
}

# Show logs
show_logs() {
    local service="${1:-megamind-mcp-server-http}"
    
    log_info "Showing logs for $service..."
    
    cd "$PROJECT_DIR"
    docker compose logs -f "$service"
}

# Health check
check_service_health() {
    log_info "Running health check..."
    
    # Check if health check script exists
    local health_script="$SCRIPT_DIR/container-health-check.py"
    
    if [[ -f "$health_script" ]]; then
        # Try to run health check with Python
        if command -v python3 &> /dev/null; then
            if python3 "$health_script" --check full --format text --exit-code; then
                log_success "Health check passed"
                return 0
            else
                log_error "Health check failed"
                return 1
            fi
        else
            log_warning "Python3 not available, using basic curl health check"
        fi
    fi
    
    # Fallback to basic curl check
    local http_port="${HTTP_PORT:-8080}"
    local health_url="http://localhost:$http_port/mcp/health"
    
    if curl -f -s "$health_url" > /dev/null; then
        log_success "Basic health check passed"
        return 0
    else
        log_error "Basic health check failed - server not responding"
        return 1
    fi
}

# Clean up
cleanup() {
    log_info "Cleaning up containers and volumes..."
    
    cd "$PROJECT_DIR"
    
    # Stop and remove containers
    docker compose down -v
    
    # Remove dangling images
    docker image prune -f
    
    log_success "Cleanup completed"
}

# Start legacy stdio server
start_legacy() {
    log_info "Starting legacy stdio server..."
    
    cd "$PROJECT_DIR"
    docker compose --profile legacy up -d
    
    log_success "Legacy services started"
}

# Main function
main() {
    local command="start"
    local env_file="$DEFAULT_ENV_FILE"
    local profile=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--env)
                env_file="$2"
                shift 2
                ;;
            -p|--profile)
                profile="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            start|stop|restart|status|logs|health|build|clean|legacy)
                command="$1"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    # Change to project directory
    cd "$PROJECT_DIR"
    
    # Setup environment (except for help and some commands)
    if [[ "$command" != "clean" ]]; then
        if ! setup_environment "$env_file"; then
            exit 1
        fi
    fi
    
    # Run pre-deployment checks (except for status, logs, clean)
    if [[ "$command" =~ ^(start|restart|build)$ ]]; then
        if ! pre_deployment_checks; then
            exit 1
        fi
    fi
    
    # Execute command
    case "$command" in
        start)
            start_services "$profile"
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs "${2:-megamind-mcp-server-http}"
            ;;
        health)
            check_service_health
            ;;
        build)
            build_containers
            ;;
        clean)
            cleanup
            ;;
        legacy)
            start_legacy
            ;;
        *)
            log_error "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"