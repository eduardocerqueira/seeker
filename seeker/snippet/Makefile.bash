#date: 2025-03-19T17:08:15Z
#url: https://api.github.com/gists/3a88a80f19cca73f9bf028c75d4a2d23
#owner: https://api.github.com/users/ahhzaky

.PHONY: build clean test lint run install cross-build

# Build variables
BINARY_NAME=NAMEBINARY
MAIN_PKG=LOCATIONFILEMAIN.GO
VERSION=$(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
BUILD_TIME=$(shell date -u '+%Y-%m-%d %H:%M:%S')
LDFLAGS=-ldflags "-s -w -X main.version=$(VERSION) -X main.buildTime=$(BUILD_TIME)"

# Default target
all: clean build

# Build the application
build:
	@echo "Building $(BINARY_NAME)..."
	@go build $(LDFLAGS) -o $(BINARY_NAME) $(MAIN_PKG)
	@echo "Build complete!"

# Clean build artifacts
clean:
	@echo "Cleaning..."
	@rm -f $(BINARY_NAME)
	@go clean
	@echo "Clean complete!"

# Run tests
test:
	@echo "Running tests..."
	@go test -v ./...

# Run static code analysis
lint:
	@echo "Running linter..."
	@if command -v golangci-lint >/dev/null 2>&1; then \
		golangci-lint run ./...; \
	else \
		echo "golangci-lint not installed, skipping"; \
	fi

# Run the application with default arguments
run: build
	@echo "Running $(BINARY_NAME)..."
	@./$(BINARY_NAME)

# Install the application
install:
	@echo "Installing $(BINARY_NAME)..."
	@go install $(LDFLAGS) $(MAIN_PKG)
	@echo "Installation complete!"

# Cross-compile for multiple platforms
cross-build:
	@echo "Cross-compiling for multiple platforms..."
	@GOOS=linux GOARCH=amd64 go build $(LDFLAGS) -o $(BINARY_NAME)-linux-amd64 $(MAIN_PKG)
	@GOOS=windows GOARCH=amd64 go build $(LDFLAGS) -o $(BINARY_NAME)-windows-amd64.exe $(MAIN_PKG)
	@GOOS=darwin GOARCH=amd64 go build $(LDFLAGS) -o $(BINARY_NAME)-darwin-amd64 $(MAIN_PKG)
	@echo "Cross-compilation complete!"

# Initialize a new module
init:
	@echo "Initializing Go module..."
	@go mod init github.com/yourusername/stagno-tool
	@go mod tidy
	@echo "Initialization complete!"

# Help command
help:
	@echo "Available targets:"
	@echo "  make            - Build the application"
	@echo "  make build      - Build the application"
	@echo "  make clean      - Remove build artifacts"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linter"
	@echo "  make run        - Build and run the application"
	@echo "  make install    - Install the application"
	@echo "  make cross-build - Cross-compile for multiple platforms"
	@echo "  make init       - Initialize a new Go module"
	@echo "  make help       - Show this help"