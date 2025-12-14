#!/bin/bash
# Stage 2 NLP Processing Service - Management Script
# Utility for managing Docker services with Docker Compose v2

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project name
PROJECT_NAME="nlp-stage2"

# Functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_env() {
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Creating from .env.example..."
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_info "Please edit .env and add your HUGGINGFACE_TOKEN"
            exit 1
        else
            print_error ".env.example not found!"
            exit 1
        fi
    fi
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi

    if ! docker compose version &> /dev/null; then
        print_error "Docker Compose v2 is not available. Please upgrade Docker."
        exit 1
    fi
}

# Command functions
cmd_start() {
    check_env
    check_docker

    print_info "Starting Stage 2 NLP Processing Service..."

    if [ "$1" == "dev" ]; then
        print_info "Starting in development mode..."
        docker compose -p ${PROJECT_NAME} up -d
    else
        print_info "Starting in production mode..."
        docker compose -p ${PROJECT_NAME} up -d
    fi

    print_success "Services started successfully!"
    print_info "Orchestrator API: http://localhost:8000"
    print_info "NER Service: http://localhost:8001"
    print_info "DP Service: http://localhost:8002"
    print_info "Event LLM Service: http://localhost:8003"
    print_info ""
    print_info "Run './run.sh logs' to view logs"
    print_info "Run './run.sh status' to check service health"
}

cmd_stop() {
    print_info "Stopping services..."
    docker compose -p ${PROJECT_NAME} stop
    print_success "Services stopped"
}

cmd_down() {
    print_info "Stopping and removing containers..."
    docker compose -p ${PROJECT_NAME} down
    print_success "Services removed"
}

cmd_restart() {
    print_info "Restarting services..."
    docker compose -p ${PROJECT_NAME} restart
    print_success "Services restarted"
}

cmd_build() {
    check_env
    check_docker

    print_info "Building Docker images..."

    if [ -n "$1" ]; then
        print_info "Building service: $1"
        docker compose -p ${PROJECT_NAME} build $1
    else
        print_info "Building all services..."
        docker compose -p ${PROJECT_NAME} build
    fi

    print_success "Build complete"
}

cmd_rebuild() {
    print_info "Rebuilding service: ${1:-all}"

    if [ "$1" == "dev" ]; then
        docker compose -p ${PROJECT_NAME} up -d --build
    elif [ -n "$1" ]; then
        docker compose -p ${PROJECT_NAME} up -d --build $1
    else
        docker compose -p ${PROJECT_NAME} up -d --build
    fi

    print_success "Rebuild complete"
}

cmd_rebuild_no_cache() {
    print_info "Rebuilding without cache: ${1:-all}"

    if [ -n "$1" ]; then
        docker compose -p ${PROJECT_NAME} build --no-cache $1
    else
        docker compose -p ${PROJECT_NAME} build --no-cache
    fi

    print_success "Rebuild complete"
}

cmd_logs() {
    if [ -n "$1" ]; then
        print_info "Showing logs for: $1"
        docker compose -p ${PROJECT_NAME} logs -f $1
    else
        print_info "Showing logs for all services..."
        docker compose -p ${PROJECT_NAME} logs -f
    fi
}

cmd_status() {
    print_info "Service status:"
    docker compose -p ${PROJECT_NAME} ps

    echo ""
    print_info "Health checks:"

    services=("orchestrator:8000" "ner-service:8001" "dp-service:8002" "event-llm-service:8003")

    for service in "${services[@]}"; do
        IFS=':' read -r name port <<< "$service"
        if curl -s -f http://localhost:${port}/health > /dev/null 2>&1; then
            print_success "${name} (port ${port}): HEALTHY"
        else
            print_error "${name} (port ${port}): UNHEALTHY"
        fi
    done
}

cmd_clean() {
    print_warning "This will remove all containers, volumes, and logs. Continue? (y/N)"
    read -r response

    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_info "Cleaning up..."
        docker compose -p ${PROJECT_NAME} down -v
        rm -rf logs/*
        print_success "Cleanup complete"
    else
        print_info "Cancelled"
    fi
}

cmd_shell() {
    service=${1:-orchestrator}
    print_info "Opening shell in ${service}..."
    docker compose -p ${PROJECT_NAME} exec ${service} /bin/bash
}

cmd_cli() {
    shift
    print_info "Running CLI command..."
    docker compose -p ${PROJECT_NAME} exec orchestrator python -m src.cli.main "$@"
}

cmd_test() {
    print_info "Running health checks on all services..."

    # Test orchestrator
    print_info "Testing orchestrator..."
    curl -X GET http://localhost:8000/health | jq '.'

    # Test NER
    print_info "Testing NER service..."
    curl -X POST http://localhost:8001/extract \
        -H "Content-Type: application/json" \
        -d '{"text":"Donald Trump met with Israeli PM in Washington.","document_id":"test"}' | jq '.entities[0]'

    print_success "Tests complete"
}

cmd_help() {
    cat << EOF
Stage 2 NLP Processing Service - Management Script

Usage: ./run.sh <command> [options]

Commands:
    start [dev]          Start all services (add 'dev' for development mode)
    stop                 Stop all services
    down                 Stop and remove all containers
    restart              Restart all services

    build [service]      Build Docker images (optionally specify service)
    rebuild [service]    Rebuild and restart service
    rebuild-no-cache     Rebuild without cache

    logs [service]       Show logs (optionally filter by service)
    status               Show service status and health

    shell [service]      Open shell in container (default: orchestrator)
    cli <args>           Run CLI commands

    test                 Run quick health tests
    clean                Remove all containers, volumes, and logs

    help                 Show this help message

Services:
    - orchestrator       Main API and coordination (port 8000)
    - ner-service        Named Entity Recognition (port 8001)
    - dp-service         Dependency Parsing (port 8002)
    - event-llm-service  Event extraction with LLM (port 8003)
    - celery-worker      Background batch processing
    - redis              Message broker and cache

Examples:
    ./run.sh start                    # Start all services
    ./run.sh logs orchestrator        # View orchestrator logs
    ./run.sh rebuild ner-service      # Rebuild NER service
    ./run.sh cli documents process "Sample text"
    ./run.sh status                   # Check all services

EOF
}

# Main script
case "${1:-help}" in
    start)
        cmd_start $2
        ;;
    stop)
        cmd_stop
        ;;
    down)
        cmd_down
        ;;
    restart)
        cmd_restart
        ;;
    build)
        cmd_build $2
        ;;
    rebuild)
        cmd_rebuild $2
        ;;
    rebuild-no-cache)
        cmd_rebuild_no_cache $2
        ;;
    logs)
        cmd_logs $2
        ;;
    status)
        cmd_status
        ;;
    clean)
        cmd_clean
        ;;
    shell)
        cmd_shell $2
        ;;
    cli)
        cmd_cli "$@"
        ;;
    test)
        cmd_test
        ;;
    help|--help|-h)
        cmd_help
        ;;
    *)
        print_error "Unknown command: $1"
        print_info "Run './run.sh help' for usage information"
        exit 1
        ;;
esac
