#!/bin/bash

# =====================================================================
# MegaMind Context Database - Production Deployment Script
# =====================================================================
# Deploy complete realm-aware database system for production use
# Created: 2025-07-12
# Usage: ./deploy_production_environment.sh [environment]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${1:-production}"
LOG_FILE="/tmp/megamind_deployment_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log "Checking deployment prerequisites..."
    
    # Check if MySQL is available
    if ! command -v mysql &> /dev/null; then
        error "MySQL client not found. Please install MySQL client."
    fi
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        error "Python 3 not found. Please install Python 3.8+."
    fi
    
    # Check for required environment variables
    if [[ -z "${DATABASE_PASSWORD:-}" ]]; then
        error "DATABASE_PASSWORD environment variable not set"
    fi
    
    if [[ -z "${DATABASE_ADMIN_PASSWORD:-}" ]]; then
        error "DATABASE_ADMIN_PASSWORD environment variable not set"
    fi
    
    success "Prerequisites check passed"
}

# Database deployment
deploy_database() {
    log "Deploying MegaMind Context Database schema..."
    
    local db_host="${DATABASE_HOST:-localhost}"
    local db_port="${DATABASE_PORT:-3306}"
    local db_name="${DATABASE_NAME:-megamind_database}"
    local db_user="${DATABASE_USER:-root}"
    
    # Create database if it doesn't exist
    mysql -h "$db_host" -P "$db_port" -u "$db_user" -p"${DATABASE_ADMIN_PASSWORD}" -e "CREATE DATABASE IF NOT EXISTS $db_name CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;" || error "Failed to create database"
    
    # Deploy schema files in order
    local schema_files=(
        "01_realm_tables.sql"
        "02_enhanced_core_tables.sql"
        "03_realm_session_tables.sql"
        "04_indexes_and_views.sql"
        "05_initial_data.sql"
        "06_inheritance_resolution.sql"
        "07_knowledge_promotion.sql"
        "08_production_deployment.sql"
        "09_global_realm_standards.sql"
        "10_initial_project_realms.sql"
    )
    
    for schema_file in "${schema_files[@]}"; do
        local file_path="$PROJECT_ROOT/database/realm_system/$schema_file"
        if [[ -f "$file_path" ]]; then
            log "Executing $schema_file..."
            mysql -h "$db_host" -P "$db_port" -u "$db_user" -p"${DATABASE_ADMIN_PASSWORD}" "$db_name" < "$file_path" || error "Failed to execute $schema_file"
        else
            warning "Schema file not found: $file_path"
        fi
    done
    
    success "Database schema deployed successfully"
}

# Create database users
setup_database_users() {
    log "Setting up database users and permissions..."
    
    local db_host="${DATABASE_HOST:-localhost}"
    local db_port="${DATABASE_PORT:-3306}"
    local db_name="${DATABASE_NAME:-megamind_database}"
    local admin_user="${DATABASE_USER:-root}"
    
    # Create application user
    mysql -h "$db_host" -P "$db_port" -u "$admin_user" -p"${DATABASE_ADMIN_PASSWORD}" <<EOF
-- Create application user
CREATE USER IF NOT EXISTS 'megamind_user'@'%' IDENTIFIED BY '${DATABASE_PASSWORD}';
GRANT SELECT, INSERT, UPDATE, DELETE ON ${db_name}.* TO 'megamind_user'@'%';

-- Create read-only user for reporting
CREATE USER IF NOT EXISTS 'megamind_readonly'@'%' IDENTIFIED BY '${DATABASE_READONLY_PASSWORD:-readonly123}';
GRANT SELECT ON ${db_name}.* TO 'megamind_readonly'@'%';

-- Create admin user for global realm management
CREATE USER IF NOT EXISTS 'megamind_admin'@'%' IDENTIFIED BY '${DATABASE_ADMIN_PASSWORD}';
GRANT ALL PRIVILEGES ON ${db_name}.* TO 'megamind_admin'@'%';

FLUSH PRIVILEGES;
EOF
    
    success "Database users configured"
}

# Install Python dependencies
install_dependencies() {
    log "Installing Python dependencies..."
    
    cd "$PROJECT_ROOT"
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install/upgrade pip
    pip install --upgrade pip
    
    # Install dependencies
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
    else
        # Install core dependencies
        pip install mysql-connector-python mcp asyncio logging
    fi
    
    success "Dependencies installed"
}

# Deploy MCP server configurations
deploy_mcp_configs() {
    log "Deploying MCP server configurations..."
    
    local config_dir="$PROJECT_ROOT/config"
    local deployment_config="$config_dir/production_mcp_configs.json"
    
    if [[ ! -f "$deployment_config" ]]; then
        error "Production MCP configuration not found: $deployment_config"
    fi
    
    # Copy configuration with environment substitution
    local target_config="/etc/megamind/mcp_servers.json"
    
    # Create directory if it doesn't exist
    sudo mkdir -p "$(dirname "$target_config")" || error "Failed to create config directory"
    
    # Process environment variables in config
    envsubst < "$deployment_config" | sudo tee "$target_config" > /dev/null || error "Failed to deploy MCP configuration"
    
    success "MCP server configurations deployed to $target_config"
}

# Set up systemd services
setup_systemd_services() {
    log "Setting up systemd services for MCP servers..."
    
    local services=(
        "megamind-ecommerce"
        "megamind-analytics"
        "megamind-mobile"
        "megamind-devops"
        "megamind-global"
    )
    
    for service in "${services[@]}"; do
        cat > "/tmp/${service}.service" <<EOF
[Unit]
Description=MegaMind Context Database MCP Server - ${service}
After=network.target mysql.service
Requires=mysql.service

[Service]
Type=simple
User=megamind
Group=megamind
WorkingDirectory=$PROJECT_ROOT
Environment=PYTHONPATH=$PROJECT_ROOT
EnvironmentFile=/etc/megamind/${service}.env
ExecStart=$PROJECT_ROOT/venv/bin/python mcp_server/context_database_server.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
        
        # Install service file
        sudo mv "/tmp/${service}.service" "/etc/systemd/system/"
        
        # Create environment file
        sudo mkdir -p "/etc/megamind"
        cat > "/tmp/${service}.env" <<EOF
MEGAMIND_PROJECT_REALM=${service/megamind-/PROJ_}
DATABASE_HOST=${DATABASE_HOST:-localhost}
DATABASE_PORT=${DATABASE_PORT:-3306}
DATABASE_NAME=${DATABASE_NAME:-megamind_database}
DATABASE_USER=megamind_user
DATABASE_PASSWORD=${DATABASE_PASSWORD}
LOG_LEVEL=INFO
CACHE_TTL=3600
MAX_SEARCH_RESULTS=50
EOF
        sudo mv "/tmp/${service}.env" "/etc/megamind/"
    done
    
    # Reload systemd and enable services
    sudo systemctl daemon-reload
    
    for service in "${services[@]}"; do
        sudo systemctl enable "${service}.service"
    done
    
    success "Systemd services configured"
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    local db_host="${DATABASE_HOST:-localhost}"
    local db_port="${DATABASE_PORT:-3306}"
    local db_name="${DATABASE_NAME:-megamind_database}"
    local db_user="megamind_user"
    
    # Test database connection
    if mysql -h "$db_host" -P "$db_port" -u "$db_user" -p"${DATABASE_PASSWORD}" -e "USE $db_name; SELECT 'Database connection successful' as status;" &>/dev/null; then
        success "Database connection verified"
    else
        error "Database connection failed"
    fi
    
    # Check schema deployment
    local table_count
    table_count=$(mysql -h "$db_host" -P "$db_port" -u "$db_user" -p"${DATABASE_PASSWORD}" "$db_name" -N -e "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = '$db_name' AND table_name LIKE 'megamind_%';" 2>/dev/null || echo "0")
    
    if [[ "$table_count" -ge 15 ]]; then
        success "Schema verification passed ($table_count tables found)"
    else
        error "Schema verification failed (only $table_count tables found, expected 15+)"
    fi
    
    # Check sample data
    local global_chunks
    global_chunks=$(mysql -h "$db_host" -P "$db_port" -u "$db_user" -p"${DATABASE_PASSWORD}" "$db_name" -N -e "SELECT COUNT(*) FROM megamind_chunks WHERE realm_id = 'GLOBAL';" 2>/dev/null || echo "0")
    
    if [[ "$global_chunks" -gt 0 ]]; then
        success "Global realm data verified ($global_chunks chunks found)"
    else
        warning "No global realm data found"
    fi
    
    # Test Python imports
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    if python -c "import mcp_server.context_database_server; print('MCP server imports successful')" &>/dev/null; then
        success "Python imports verified"
    else
        error "Python import verification failed"
    fi
    
    success "Deployment verification completed"
}

# Start services
start_services() {
    log "Starting MCP services..."
    
    local services=(
        "megamind-global"      # Start global first
        "megamind-ecommerce"
        "megamind-analytics"
        "megamind-mobile"
        "megamind-devops"
    )
    
    for service in "${services[@]}"; do
        log "Starting ${service}..."
        sudo systemctl start "${service}.service"
        
        # Wait for service to start
        sleep 5
        
        if sudo systemctl is-active --quiet "${service}.service"; then
            success "${service} started successfully"
        else
            error "${service} failed to start"
        fi
    done
    
    success "All MCP services started"
}

# Generate deployment report
generate_report() {
    log "Generating deployment report..."
    
    local report_file="/tmp/megamind_deployment_report_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$report_file" <<EOF
MegaMind Context Database - Production Deployment Report
========================================================

Deployment Date: $(date)
Environment: $ENVIRONMENT
Log File: $LOG_FILE

Database Configuration:
- Host: ${DATABASE_HOST:-localhost}
- Port: ${DATABASE_PORT:-3306}
- Database: ${DATABASE_NAME:-megamind_database}

Realm Configuration:
- Global Realm: GLOBAL (organizational standards)
- Project Realms: PROJ_ECOMMERCE, PROJ_ANALYTICS, PROJ_MOBILE, PROJ_DEVOPS

MCP Server Instances:
- megamind-global: Global organizational standards
- megamind-ecommerce: E-commerce platform knowledge
- megamind-analytics: Data analytics pipeline knowledge
- megamind-mobile: Mobile application knowledge
- megamind-devops: DevOps infrastructure knowledge

Service Status:
EOF
    
    # Add service status to report
    for service in megamind-global megamind-ecommerce megamind-analytics megamind-mobile megamind-devops; do
        if sudo systemctl is-active --quiet "${service}.service" 2>/dev/null; then
            echo "- $service: RUNNING" >> "$report_file"
        else
            echo "- $service: STOPPED/FAILED" >> "$report_file"
        fi
    done
    
    cat >> "$report_file" <<EOF

Next Steps:
1. Configure client applications to use MCP servers
2. Set up monitoring and alerting
3. Configure backup procedures
4. Review security settings
5. Train users on realm-based knowledge organization

Configuration Files:
- MCP Config: /etc/megamind/mcp_servers.json
- Service Files: /etc/systemd/system/megamind-*.service
- Environment Files: /etc/megamind/*.env

For support and documentation, see:
- Project Documentation: $PROJECT_ROOT/README.md
- Session Notes: $PROJECT_ROOT/.claude/sessions/
- Database Schema: $PROJECT_ROOT/database/realm_system/
EOF
    
    success "Deployment report generated: $report_file"
    cat "$report_file"
}

# Main deployment flow
main() {
    log "Starting MegaMind Context Database production deployment..."
    log "Environment: $ENVIRONMENT"
    log "Project Root: $PROJECT_ROOT"
    
    check_prerequisites
    deploy_database
    setup_database_users
    install_dependencies
    deploy_mcp_configs
    setup_systemd_services
    verify_deployment
    start_services
    generate_report
    
    success "MegaMind Context Database deployment completed successfully!"
    success "Check the deployment report above for next steps and configuration details."
}

# Handle script arguments
case "${1:-production}" in
    "development"|"staging"|"production")
        ENVIRONMENT="$1"
        ;;
    "--help"|"-h")
        echo "Usage: $0 [environment]"
        echo "Environments: development, staging, production (default)"
        echo ""
        echo "Required environment variables:"
        echo "- DATABASE_PASSWORD: Password for application database user"
        echo "- DATABASE_ADMIN_PASSWORD: Password for admin database user"
        echo ""
        echo "Optional environment variables:"
        echo "- DATABASE_HOST: Database host (default: localhost)"
        echo "- DATABASE_PORT: Database port (default: 3306)"
        echo "- DATABASE_NAME: Database name (default: megamind_database)"
        echo "- DATABASE_USER: Admin database user (default: root)"
        exit 0
        ;;
    *)
        error "Invalid environment: $1. Use development, staging, or production."
        ;;
esac

# Run main deployment
main