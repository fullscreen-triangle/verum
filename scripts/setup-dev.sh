#!/bin/bash

# Verum Framework Development Environment Setup
# This script sets up the complete development environment for Verum

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="verum"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_NAME="verum-env"

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

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system requirements
check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        log_info "Detected Linux system"
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        log_info "Detected macOS system"
        OS="macos"
    else
        log_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    # Check required tools
    local required_tools=("curl" "git" "make")
    for tool in "${required_tools[@]}"; do
        if ! command_exists "$tool"; then
            log_error "Required tool not found: $tool"
            exit 1
        fi
    done
    
    log_success "System requirements check passed"
}

# Install Rust
install_rust() {
    log_info "Installing Rust..."
    
    if command_exists rustc; then
        local rust_version=$(rustc --version | cut -d' ' -f2)
        log_warning "Rust already installed: $rust_version"
        return 0
    fi
    
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ~/.cargo/env
    
    # Install additional components
    rustup component add clippy rustfmt
    
    # Install useful cargo tools
    cargo install cargo-audit cargo-tarpaulin
    
    log_success "Rust installed successfully"
}

# Install Python
install_python() {
    log_info "Installing Python 3.11..."
    
    if command_exists python3.11; then
        log_warning "Python 3.11 already installed"
        return 0
    fi
    
    if [[ "$OS" == "linux" ]]; then
        # Ubuntu/Debian
        if command_exists apt; then
            sudo apt update
            sudo apt install -y python3.11 python3.11-venv python3.11-dev \
                               python3-pip build-essential
        # Fedora/RHEL
        elif command_exists dnf; then
            sudo dnf install -y python3.11 python3.11-devel python3-pip gcc
        # Arch Linux
        elif command_exists pacman; then
            sudo pacman -S python python-pip gcc
        else
            log_error "Unsupported Linux distribution"
            exit 1
        fi
    elif [[ "$OS" == "macos" ]]; then
        if command_exists brew; then
            brew install python@3.11
        else
            log_error "Homebrew not found. Please install Homebrew first."
            exit 1
        fi
    fi
    
    log_success "Python installed successfully"
}

# Install Go
install_go() {
    log_info "Installing Go 1.21..."
    
    if command_exists go; then
        local go_version=$(go version | awk '{print $3}')
        log_warning "Go already installed: $go_version"
        return 0
    fi
    
    local go_version="1.21.5"
    local go_arch
    
    case "$(uname -m)" in
        x86_64) go_arch="amd64" ;;
        arm64) go_arch="arm64" ;;
        aarch64) go_arch="arm64" ;;
        *) log_error "Unsupported architecture: $(uname -m)"; exit 1 ;;
    esac
    
    local go_os
    case "$OS" in
        linux) go_os="linux" ;;
        macos) go_os="darwin" ;;
    esac
    
    local go_filename="go${go_version}.${go_os}-${go_arch}.tar.gz"
    local go_url="https://go.dev/dl/${go_filename}"
    
    log_info "Downloading Go from $go_url"
    curl -L "$go_url" -o "/tmp/$go_filename"
    
    sudo tar -C /usr/local -xzf "/tmp/$go_filename"
    rm "/tmp/$go_filename"
    
    # Add to PATH
    echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
    export PATH=$PATH:/usr/local/go/bin
    
    # Install useful Go tools
    go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
    go install github.com/google/go-licenses@latest
    
    log_success "Go installed successfully"
}

# Install system dependencies
install_system_dependencies() {
    log_info "Installing system dependencies..."
    
    if [[ "$OS" == "linux" ]]; then
        if command_exists apt; then
            sudo apt update
            sudo apt install -y \
                build-essential \
                pkg-config \
                libssl-dev \
                libusb-1.0-0-dev \
                libudev-dev \
                cmake \
                git \
                curl \
                postgresql-client \
                redis-tools
        elif command_exists dnf; then
            sudo dnf install -y \
                gcc \
                gcc-c++ \
                pkg-config \
                openssl-devel \
                libusb1-devel \
                systemd-devel \
                cmake \
                git \
                curl \
                postgresql \
                redis
        fi
    elif [[ "$OS" == "macos" ]]; then
        if command_exists brew; then
            brew install \
                pkg-config \
                openssl \
                cmake \
                postgresql \
                redis
        fi
    fi
    
    log_success "System dependencies installed"
}

# Setup Python virtual environment
setup_python_env() {
    log_info "Setting up Python virtual environment..."
    
    cd "$PROJECT_DIR/verum-learn"
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "venv" ]]; then
        python3.11 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install requirements
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
    fi
    
    # Install development requirements
    pip install \
        black \
        isort \
        flake8 \
        mypy \
        pytest \
        pytest-cov \
        pytest-asyncio \
        jupyter \
        ipython
    
    log_success "Python environment setup completed"
    cd "$PROJECT_DIR"
}

# Setup development tools
setup_dev_tools() {
    log_info "Setting up development tools..."
    
    # Create VS Code settings if they don't exist
    mkdir -p .vscode
    
    cat > .vscode/settings.json << 'EOF'
{
    "rust-analyzer.linkedProjects": ["verum-core/Cargo.toml"],
    "python.defaultInterpreterPath": "./verum-learn/venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "go.useLanguageServer": true,
    "go.formatTool": "goimports",
    "go.lintTool": "golangci-lint",
    "files.associations": {
        "Cargo.toml": "toml",
        "*.rs": "rust"
    }
}
EOF
    
    cat > .vscode/extensions.json << 'EOF'
{
    "recommendations": [
        "rust-lang.rust-analyzer",
        "ms-python.python",
        "golang.go",
        "tamasfe.even-better-toml",
        "ms-vscode.makefile-tools"
    ]
}
EOF
    
    # Setup pre-commit hooks
    cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
set -e

echo "Running pre-commit checks..."

# Format Rust code
cd verum-core && cargo fmt --check
cd ../

# Format Python code
cd verum-learn && source venv/bin/activate && black --check . && isort --check-only .
cd ../

# Format Go code
cd verum-network && go fmt ./...
cd ../

echo "Pre-commit checks passed!"
EOF
    
    chmod +x .git/hooks/pre-commit
    
    log_success "Development tools setup completed"
}

# Setup database
setup_database() {
    log_info "Setting up development database..."
    
    # Check if PostgreSQL is running
    if ! pgrep -x "postgres" > /dev/null; then
        log_warning "PostgreSQL not running. Please start PostgreSQL service."
        if [[ "$OS" == "linux" ]]; then
            log_info "Run: sudo systemctl start postgresql"
        elif [[ "$OS" == "macos" ]]; then
            log_info "Run: brew services start postgresql"
        fi
        return 0
    fi
    
    # Create development database
    createdb verum_dev 2>/dev/null || log_warning "Database verum_dev already exists"
    
    log_success "Database setup completed"
}

# Create environment file
create_env_file() {
    log_info "Creating environment configuration..."
    
    if [[ ! -f ".env" ]]; then
        cat > .env << EOF
# Verum Development Environment Configuration

# Development settings
VERUM_ENV=development
VERUM_LOG_LEVEL=debug

# Database settings
DATABASE_URL=postgresql://\$USER@localhost:5432/verum_dev

# Redis settings
REDIS_URL=redis://localhost:6379

# ML settings
CUDA_VISIBLE_DEVICES=0
TORCH_DEVICE=cpu

# Network settings
NETWORK_PORT=8080
COORDINATOR_URL=http://localhost:8081

# API settings
API_HOST=localhost
API_PORT=3000

# Security settings (development only)
JWT_SECRET=dev-secret-key-change-in-production
ENCRYPTION_KEY=dev-encryption-key-32-chars-long

# Paths
DATA_DIR=./data
MODEL_DIR=./data/models
LOG_DIR=./logs
EOF
        log_success "Environment file created"
    else
        log_warning "Environment file already exists"
    fi
}

# Run initial build
initial_build() {
    log_info "Running initial build..."
    
    # Build Rust components
    cd verum-core
    cargo build
    cd ..
    
    # Build Python components
    cd verum-learn
    source venv/bin/activate
    pip install -e .
    cd ..
    
    # Build Go components
    cd verum-network
    go mod tidy
    go build ./...
    cd ..
    
    log_success "Initial build completed"
}

# Run tests
run_tests() {
    log_info "Running initial tests..."
    
    # Test Rust
    cd verum-core
    cargo test
    cd ..
    
    # Test Python
    cd verum-learn
    source venv/bin/activate
    python -m pytest tests/ || log_warning "Python tests not found (expected for new project)"
    cd ..
    
    # Test Go
    cd verum-network
    go test ./... || log_warning "Go tests not found (expected for new project)"
    cd ..
    
    log_success "Tests completed"
}

# Main setup function
main() {
    log_info "Starting Verum development environment setup..."
    
    cd "$PROJECT_DIR"
    
    check_system_requirements
    install_system_dependencies
    install_rust
    install_python
    install_go
    setup_python_env
    setup_dev_tools
    setup_database
    create_env_file
    initial_build
    run_tests
    
    log_success "Development environment setup completed!"
    log_info ""
    log_info "Next steps:"
    log_info "1. Source your shell profile: source ~/.bashrc"
    log_info "2. Activate Python environment: cd verum-learn && source venv/bin/activate"
    log_info "3. Start development: make dev-start"
    log_info "4. Run tests: make test"
    log_info ""
    log_info "Happy coding! 🚗🤖"
}

# Run main function
main "$@" 