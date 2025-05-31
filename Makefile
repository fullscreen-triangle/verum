# Verum Framework Build System
# Multi-language project: Rust (verum-core), Python (verum-learn), Go (verum-network)

.PHONY: all clean build test install help
.DEFAULT_GOAL := help

# Colors for output
RED    := \033[31m
GREEN  := \033[32m
YELLOW := \033[33m
BLUE   := \033[34m
RESET  := \033[0m

# Project information
PROJECT_NAME := verum
VERSION := $(shell git describe --tags --always --dirty)
BUILD_TIME := $(shell date -u '+%Y-%m-%d_%H:%M:%S')

# Directories
CORE_DIR := verum-core
LEARN_DIR := verum-learn
NETWORK_DIR := verum-network
SCRIPTS_DIR := scripts
DOCS_DIR := docs
TESTS_DIR := tests

# Build targets
RUST_TARGET := target/release/verum-core
PYTHON_TARGET := $(LEARN_DIR)/dist
GO_TARGET := $(NETWORK_DIR)/bin

help: ## Show this help message
	@echo "$(GREEN)Verum Framework Build System$(RESET)"
	@echo "$(BLUE)Version: $(VERSION)$(RESET)"
	@echo ""
	@echo "$(YELLOW)Available targets:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# =============================================================================
# Main targets
# =============================================================================

all: build ## Build all components

build: build-core build-learn build-network ## Build all components

clean: clean-core clean-learn clean-network ## Clean all build artifacts

test: test-core test-learn test-network ## Run all tests

install: install-deps build ## Install dependencies and build

# =============================================================================
# Rust (verum-core) targets
# =============================================================================

build-core: ## Build Rust core component
	@echo "$(BLUE)Building verum-core (Rust)...$(RESET)"
	cd $(CORE_DIR) && cargo build --release
	@echo "$(GREEN)✓ verum-core built successfully$(RESET)"

test-core: ## Run Rust tests
	@echo "$(BLUE)Running Rust tests...$(RESET)"
	cd $(CORE_DIR) && cargo test --release
	@echo "$(GREEN)✓ Rust tests passed$(RESET)"

clean-core: ## Clean Rust build artifacts
	@echo "$(BLUE)Cleaning Rust artifacts...$(RESET)"
	cd $(CORE_DIR) && cargo clean

bench-core: ## Run Rust benchmarks
	@echo "$(BLUE)Running Rust benchmarks...$(RESET)"
	cd $(CORE_DIR) && cargo bench

clippy-core: ## Run Rust linter
	@echo "$(BLUE)Running Rust clippy...$(RESET)"
	cd $(CORE_DIR) && cargo clippy --all-targets --all-features -- -D warnings

fmt-core: ## Format Rust code
	@echo "$(BLUE)Formatting Rust code...$(RESET)"
	cd $(CORE_DIR) && cargo fmt

doc-core: ## Generate Rust documentation
	@echo "$(BLUE)Generating Rust documentation...$(RESET)"
	cd $(CORE_DIR) && cargo doc --no-deps --open

# =============================================================================
# Python (verum-learn) targets
# =============================================================================

build-learn: ## Build Python learning component
	@echo "$(BLUE)Building verum-learn (Python)...$(RESET)"
	cd $(LEARN_DIR) && python -m build
	@echo "$(GREEN)✓ verum-learn built successfully$(RESET)"

test-learn: ## Run Python tests
	@echo "$(BLUE)Running Python tests...$(RESET)"
	cd $(LEARN_DIR) && python -m pytest tests/ -v
	@echo "$(GREEN)✓ Python tests passed$(RESET)"

clean-learn: ## Clean Python build artifacts
	@echo "$(BLUE)Cleaning Python artifacts...$(RESET)"
	cd $(LEARN_DIR) && rm -rf build/ dist/ *.egg-info/ .pytest_cache/ __pycache__/
	find $(LEARN_DIR) -name "*.pyc" -delete
	find $(LEARN_DIR) -name "__pycache__" -type d -exec rm -rf {} +

lint-learn: ## Run Python linting
	@echo "$(BLUE)Running Python linting...$(RESET)"
	cd $(LEARN_DIR) && python -m flake8 verum_learn/
	cd $(LEARN_DIR) && python -m black --check verum_learn/
	cd $(LEARN_DIR) && python -m isort --check-only verum_learn/

fmt-learn: ## Format Python code
	@echo "$(BLUE)Formatting Python code...$(RESET)"
	cd $(LEARN_DIR) && python -m black verum_learn/
	cd $(LEARN_DIR) && python -m isort verum_learn/

type-check-learn: ## Run Python type checking
	@echo "$(BLUE)Running Python type checking...$(RESET)"
	cd $(LEARN_DIR) && python -m mypy verum_learn/

coverage-learn: ## Run Python test coverage
	@echo "$(BLUE)Running Python test coverage...$(RESET)"
	cd $(LEARN_DIR) && python -m pytest --cov=verum_learn --cov-report=html tests/

# =============================================================================
# Go (verum-network) targets
# =============================================================================

build-network: ## Build Go network component
	@echo "$(BLUE)Building verum-network (Go)...$(RESET)"
	cd $(NETWORK_DIR) && go build -ldflags="-X main.version=$(VERSION) -X main.buildTime=$(BUILD_TIME)" -o bin/ ./cmd/...
	@echo "$(GREEN)✓ verum-network built successfully$(RESET)"

test-network: ## Run Go tests
	@echo "$(BLUE)Running Go tests...$(RESET)"
	cd $(NETWORK_DIR) && go test -v ./...
	@echo "$(GREEN)✓ Go tests passed$(RESET)"

clean-network: ## Clean Go build artifacts
	@echo "$(BLUE)Cleaning Go artifacts...$(RESET)"
	cd $(NETWORK_DIR) && rm -rf bin/ && go clean -cache -modcache

bench-network: ## Run Go benchmarks
	@echo "$(BLUE)Running Go benchmarks...$(RESET)"
	cd $(NETWORK_DIR) && go test -bench=. -benchmem ./...

lint-network: ## Run Go linting
	@echo "$(BLUE)Running Go linting...$(RESET)"
	cd $(NETWORK_DIR) && golangci-lint run

fmt-network: ## Format Go code
	@echo "$(BLUE)Formatting Go code...$(RESET)"
	cd $(NETWORK_DIR) && go fmt ./...

vet-network: ## Run Go vet
	@echo "$(BLUE)Running Go vet...$(RESET)"
	cd $(NETWORK_DIR) && go vet ./...

# =============================================================================
# Dependency management
# =============================================================================

install-deps: install-rust-deps install-python-deps install-go-deps ## Install all dependencies

install-rust-deps: ## Install Rust dependencies
	@echo "$(BLUE)Installing Rust dependencies...$(RESET)"
	cd $(CORE_DIR) && cargo fetch

install-python-deps: ## Install Python dependencies
	@echo "$(BLUE)Installing Python dependencies...$(RESET)"
	cd $(LEARN_DIR) && pip install -r requirements.txt

install-go-deps: ## Install Go dependencies
	@echo "$(BLUE)Installing Go dependencies...$(RESET)"
	cd $(NETWORK_DIR) && go mod download

update-deps: ## Update all dependencies
	@echo "$(BLUE)Updating dependencies...$(RESET)"
	cd $(CORE_DIR) && cargo update
	cd $(LEARN_DIR) && pip install --upgrade -r requirements.txt
	cd $(NETWORK_DIR) && go get -u ./... && go mod tidy

# =============================================================================
# Testing and Quality Assurance
# =============================================================================

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(RESET)"
	cd $(CORE_DIR) && cargo test --lib
	cd $(LEARN_DIR) && python -m pytest tests/unit/ -v
	cd $(NETWORK_DIR) && go test -short ./...

test-integration: ## Run integration tests
	@echo "$(BLUE)Running integration tests...$(RESET)"
	$(SCRIPTS_DIR)/run-integration-tests.sh

test-performance: ## Run performance tests
	@echo "$(BLUE)Running performance tests...$(RESET)"
	$(SCRIPTS_DIR)/run-performance-tests.sh

benchmark: bench-core bench-network ## Run all benchmarks

lint: clippy-core lint-learn lint-network ## Run all linting

fmt: fmt-core fmt-learn fmt-network ## Format all code

check: lint test ## Run all checks (lint + test)

# =============================================================================
# Development environment
# =============================================================================

dev-setup: ## Setup development environment
	@echo "$(BLUE)Setting up development environment...$(RESET)"
	$(SCRIPTS_DIR)/setup-dev.sh

dev-start: ## Start development services
	@echo "$(BLUE)Starting development services...$(RESET)"
	docker-compose up -d

dev-stop: ## Stop development services
	@echo "$(BLUE)Stopping development services...$(RESET)"
	docker-compose down

dev-logs: ## Show development service logs
	docker-compose logs -f

dev-clean: ## Clean development environment
	@echo "$(BLUE)Cleaning development environment...$(RESET)"
	docker-compose down -v
	docker system prune -f

# =============================================================================
# Documentation
# =============================================================================

docs: doc-core ## Generate all documentation
	@echo "$(BLUE)Generating documentation...$(RESET)"
	# Additional documentation generation can be added here

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(RESET)"
	cd $(DOCS_DIR) && python -m http.server 8000

# =============================================================================
# Release and deployment
# =============================================================================

release: check build ## Prepare release build
	@echo "$(GREEN)Release $(VERSION) ready!$(RESET)"
	@echo "Rust binary: $(CORE_DIR)/$(RUST_TARGET)"
	@echo "Python package: $(LEARN_DIR)/dist/"
	@echo "Go binaries: $(NETWORK_DIR)/bin/"

package: build ## Package all components
	@echo "$(BLUE)Packaging components...$(RESET)"
	mkdir -p dist/
	cp $(CORE_DIR)/$(RUST_TARGET) dist/
	cp -r $(LEARN_DIR)/dist/* dist/
	cp -r $(NETWORK_DIR)/bin/* dist/
	tar -czf dist/verum-$(VERSION).tar.gz -C dist/ .
	@echo "$(GREEN)✓ Package created: dist/verum-$(VERSION).tar.gz$(RESET)"

deploy-dev: ## Deploy to development environment
	@echo "$(BLUE)Deploying to development...$(RESET)"
	$(SCRIPTS_DIR)/deploy.sh dev

deploy-staging: ## Deploy to staging environment
	@echo "$(BLUE)Deploying to staging...$(RESET)"
	$(SCRIPTS_DIR)/deploy.sh staging

deploy-prod: ## Deploy to production environment
	@echo "$(BLUE)Deploying to production...$(RESET)"
	$(SCRIPTS_DIR)/deploy.sh prod

# =============================================================================
# Utilities
# =============================================================================

version: ## Show version information
	@echo "$(GREEN)Verum Framework$(RESET)"
	@echo "Version: $(VERSION)"
	@echo "Build time: $(BUILD_TIME)"
	@echo ""
	@echo "Components:"
	@cd $(CORE_DIR) && echo "  Rust (verum-core): $$(cargo --version)"
	@cd $(LEARN_DIR) && echo "  Python (verum-learn): $$(python --version)"
	@cd $(NETWORK_DIR) && echo "  Go (verum-network): $$(go version)"

deps-check: ## Check dependency status
	@echo "$(BLUE)Checking dependencies...$(RESET)"
	@echo "Rust dependencies:"
	@cd $(CORE_DIR) && cargo tree --depth 1
	@echo ""
	@echo "Python dependencies:"
	@cd $(LEARN_DIR) && pip list
	@echo ""
	@echo "Go dependencies:"
	@cd $(NETWORK_DIR) && go list -m all

security-check: ## Run security checks
	@echo "$(BLUE)Running security checks...$(RESET)"
	cd $(CORE_DIR) && cargo audit
	cd $(LEARN_DIR) && python -m safety check
	cd $(NETWORK_DIR) && gosec ./...

size-check: ## Check binary sizes
	@echo "$(BLUE)Checking binary sizes...$(RESET)"
	@if [ -f "$(CORE_DIR)/$(RUST_TARGET)" ]; then \
		echo "Rust binary: $$(du -h $(CORE_DIR)/$(RUST_TARGET) | cut -f1)"; \
	fi
	@if [ -d "$(NETWORK_DIR)/bin" ]; then \
		echo "Go binaries:"; \
		du -h $(NETWORK_DIR)/bin/* 2>/dev/null || true; \
	fi

# =============================================================================
# Docker targets
# =============================================================================

docker-build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(RESET)"
	docker build -t verum-core:$(VERSION) -f $(CORE_DIR)/Dockerfile $(CORE_DIR)
	docker build -t verum-learn:$(VERSION) -f $(LEARN_DIR)/Dockerfile $(LEARN_DIR)
	docker build -t verum-network:$(VERSION) -f $(NETWORK_DIR)/Dockerfile $(NETWORK_DIR)

docker-test: ## Run tests in Docker
	@echo "$(BLUE)Running tests in Docker...$(RESET)"
	docker-compose -f docker-compose.test.yml up --abort-on-container-exit

docker-clean: ## Clean Docker artifacts
	@echo "$(BLUE)Cleaning Docker artifacts...$(RESET)"
	docker system prune -f
	docker volume prune -f

# =============================================================================
# Maintenance
# =============================================================================

update-copyright: ## Update copyright headers
	@echo "$(BLUE)Updating copyright headers...$(RESET)"
	$(SCRIPTS_DIR)/update-copyright.sh

licenses-check: ## Check license compatibility
	@echo "$(BLUE)Checking license compatibility...$(RESET)"
	cd $(CORE_DIR) && cargo license
	cd $(LEARN_DIR) && pip-licenses
	cd $(NETWORK_DIR) && go-licenses report ./...

todo: ## Find TODO comments
	@echo "$(BLUE)Finding TODO comments...$(RESET)"
	@grep -r "TODO\|FIXME\|XXX\|HACK" --include="*.rs" --include="*.py" --include="*.go" . || true

stats: ## Show project statistics
	@echo "$(BLUE)Project Statistics:$(RESET)"
	@echo "Lines of code:"
	@find $(CORE_DIR)/src -name "*.rs" | xargs wc -l | tail -1 | awk '{print "  Rust: " $$1 " lines"}'
	@find $(LEARN_DIR)/verum_learn -name "*.py" | xargs wc -l | tail -1 | awk '{print "  Python: " $$1 " lines"}'
	@find $(NETWORK_DIR) -name "*.go" -not -path "*/vendor/*" | xargs wc -l | tail -1 | awk '{print "  Go: " $$1 " lines"}'
	@echo ""
	@echo "Test coverage:"
	@cd $(CORE_DIR) && cargo tarpaulin --skip-clean --out Stdout 2>/dev/null | grep "%" | tail -1 || echo "  Rust: Coverage data not available"
	@echo "  Python: Run 'make coverage-learn' for detailed coverage"
	@echo "  Go: Run 'go test -cover ./...' in verum-network for coverage"

# =============================================================================
# Special targets
# =============================================================================

# Ensure scripts are executable
$(SCRIPTS_DIR)/%: $(SCRIPTS_DIR)/%.sh
	chmod +x $<

# Create necessary directories
$(GO_TARGET) $(PYTHON_TARGET):
	mkdir -p $@

# =============================================================================
# Environment checks
# =============================================================================

check-rust:
	@which rustc >/dev/null 2>&1 || (echo "$(RED)Error: Rust not found$(RESET)" && exit 1)

check-python:
	@which python >/dev/null 2>&1 || (echo "$(RED)Error: Python not found$(RESET)" && exit 1)

check-go:
	@which go >/dev/null 2>&1 || (echo "$(RED)Error: Go not found$(RESET)" && exit 1)

check-env: check-rust check-python check-go ## Verify development environment 