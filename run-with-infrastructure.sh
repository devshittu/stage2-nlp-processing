#!/bin/bash

# =============================================================================
# Stage 2 NLP Processing Service - Infrastructure Integration Script
# =============================================================================
# This script manages Stage 2 deployment with centralized infrastructure
# Usage: ./run-with-infrastructure.sh [start|stop|restart|logs|status]
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROJECT_NAME="stage2-nlp"
COMPOSE_FILE="docker-compose.infrastructure.yml"
INFRASTRUCTURE_DIR="../infrastructure"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

print_header() {
    echo -e "${BLUE}=====================================================================${NC}"
    echo -e "${BLUE}  Stage 2 NLP Processing Service - Infrastructure Integration${NC}"
    echo -e "${BLUE}=====================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check if infrastructure is running
check_infrastructure() {
    print_info "Checking infrastructure status..."

    if [ ! -d "$INFRASTRUCTURE_DIR" ]; then
        print_error "Infrastructure directory not found: $INFRASTRUCTURE_DIR"
        exit 1
    fi

    cd "$INFRASTRUCTURE_DIR"

    # Check critical services (with storytelling- prefix)
    local services=("storytelling-redis-broker" "storytelling-redis-cache" "storytelling-postgres" "storytelling-traefik")
    local all_running=true

    for service in "${services[@]}"; do
        if docker ps --format "{{.Names}}" | grep -q "^$service$"; then
            print_success "$service is running"
        else
            print_error "$service is not running"
            all_running=false
        fi
    done

    cd - > /dev/null

    if [ "$all_running" = false ]; then
        print_error "Infrastructure is not fully running"
        print_info "Start infrastructure with: cd $INFRASTRUCTURE_DIR && ./scripts/start.sh"
        exit 1
    fi

    print_success "Infrastructure is running"
}

# Check environment file
check_env() {
    print_info "Checking environment configuration..."

    if [ ! -f .env ]; then
        print_warning ".env file not found"
        print_info "Creating .env from .env.example..."

        if [ -f .env.example ]; then
            cp .env.example .env
            print_success "Created .env file"
            print_warning "Please edit .env and set:"
            print_warning "  - STAGE2_POSTGRES_PASSWORD"
            print_warning "  - METADATA_SERVICE_PASSWORD"
            print_warning "  - HUGGINGFACE_TOKEN"
        else
            print_error ".env.example not found"
            exit 1
        fi
    else
        print_success ".env file found"

        # Check critical variables
        local missing_vars=()

        if ! grep -q "^STAGE2_POSTGRES_PASSWORD=" .env || grep -q "^STAGE2_POSTGRES_PASSWORD=your_secure_password_here" .env; then
            missing_vars+=("STAGE2_POSTGRES_PASSWORD")
        fi

        if ! grep -q "^METADATA_SERVICE_PASSWORD=" .env || grep -q "^METADATA_SERVICE_PASSWORD=metadata_secure_password" .env; then
            missing_vars+=("METADATA_SERVICE_PASSWORD")
        fi

        if ! grep -q "^HUGGINGFACE_TOKEN=" .env || grep -q "^HUGGINGFACE_TOKEN=\"hf_" .env; then
            missing_vars+=("HUGGINGFACE_TOKEN")
        fi

        if [ ${#missing_vars[@]} -gt 0 ]; then
            print_warning "Missing or default passwords detected:"
            for var in "${missing_vars[@]}"; do
                print_warning "  - $var"
            done
            print_info "Please set secure values before starting"
        fi
    fi
}

# Check shared metadata registry package
check_shared_package() {
    print_info "Checking shared-metadata-registry package..."

    if [ ! -d "$INFRASTRUCTURE_DIR/shared-metadata-registry" ]; then
        print_warning "shared-metadata-registry not found"
        print_warning "Metadata registry integration will be disabled"
        return
    fi

    print_success "shared-metadata-registry package found"
}

# Create necessary directories
create_directories() {
    print_info "Creating data directories..."

    mkdir -p data/{processed,raw,checkpoints}
    mkdir -p logs
    mkdir -p config

    print_success "Directories created"
}

# Build and install shared registry
install_shared_registry() {
    print_info "Installing shared-metadata-registry package..."

    if [ ! -d "$INFRASTRUCTURE_DIR/shared-metadata-registry" ]; then
        print_warning "Skipping registry installation (package not found)"
        return
    fi

    # Copy to temporary location for Docker build context
    print_info "Copying registry package to build context..."
    rm -rf ./.shared-metadata-registry
    cp -r "$INFRASTRUCTURE_DIR/shared-metadata-registry" ./.shared-metadata-registry

    print_success "Registry package prepared for installation"
}

# Start services
start_services() {
    print_header
    print_info "Starting Stage 2 NLP Processing Service..."

    check_infrastructure
    check_env
    check_shared_package
    create_directories
    install_shared_registry

    print_info "Building and starting containers..."
    docker compose -f "$COMPOSE_FILE" up -d --build

    print_info "Waiting for services to be healthy..."
    sleep 15

    # Check service health
    if docker compose -f "$COMPOSE_FILE" ps | grep -q "nlp-orchestrator.*running"; then
        print_success "Orchestrator started"
    else
        print_error "Orchestrator failed to start"
        docker compose -f "$COMPOSE_FILE" logs orchestrator-service
        exit 1
    fi

    if docker compose -f "$COMPOSE_FILE" ps | grep -q "nlp-celery-worker.*running"; then
        print_success "Celery worker started"
    else
        print_error "Celery worker failed to start"
        docker compose -f "$COMPOSE_FILE" logs celery-worker
        exit 1
    fi

    if docker compose -f "$COMPOSE_FILE" ps | grep -q "nlp-ner-service.*running"; then
        print_success "NER service started"
    else
        print_warning "NER service not running (may still be loading models)"
    fi

    if docker compose -f "$COMPOSE_FILE" ps | grep -q "nlp-dp-service.*running"; then
        print_success "DP service started"
    else
        print_warning "DP service not running (may still be loading models)"
    fi

    if docker compose -f "$COMPOSE_FILE" ps | grep -q "nlp-event-llm-service.*running"; then
        print_success "Event LLM service started"
    else
        print_warning "Event LLM service not running (may still be loading models)"
    fi

    print_success "Stage 2 NLP Processing Service started successfully!"
    print_info ""
    print_info "Access via Traefik: http://localhost/api/v1/nlp/health"
    print_info "View logs: ./run-with-infrastructure.sh logs"
    print_info "Check status: ./run-with-infrastructure.sh status"
}

# Stop services
stop_services() {
    print_header
    print_info "Stopping Stage 2 NLP Processing Service..."

    docker compose -f "$COMPOSE_FILE" down

    # Clean up temporary registry package
    if [ -d ./.shared-metadata-registry ]; then
        rm -rf ./.shared-metadata-registry
    fi

    print_success "Stage 2 stopped"
}

# Restart services
restart_services() {
    print_header
    print_info "Restarting Stage 2 NLP Processing Service..."

    stop_services
    sleep 2
    start_services
}

# Show logs
show_logs() {
    print_header
    print_info "Showing Stage 2 logs (Ctrl+C to exit)..."

    docker compose -f "$COMPOSE_FILE" logs -f
}

# Show status
show_status() {
    print_header
    print_info "Stage 2 NLP Processing Service Status:"
    echo ""

    docker compose -f "$COMPOSE_FILE" ps

    echo ""
    print_info "Health Check:"

    # Check health endpoint
    if curl -sf http://localhost/api/v1/nlp/health > /dev/null 2>&1; then
        print_success "API is healthy"
        curl -s http://localhost/api/v1/nlp/health | python3 -m json.tool 2>/dev/null || curl -s http://localhost/api/v1/nlp/health
    else
        print_warning "API health check failed"
        print_info "Service may still be starting up..."
        print_info "Try direct access: http://localhost:8000/health (if using standalone mode)"
    fi
}

# -----------------------------------------------------------------------------
# Main Script
# -----------------------------------------------------------------------------

case "${1:-}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    *)
        print_header
        echo "Usage: $0 {start|stop|restart|logs|status}"
        echo ""
        echo "Commands:"
        echo "  start    - Start Stage 2 with infrastructure"
        echo "  stop     - Stop Stage 2 services"
        echo "  restart  - Restart Stage 2 services"
        echo "  logs     - Show service logs (follow mode)"
        echo "  status   - Show service status and health"
        echo ""
        exit 1
        ;;
esac
