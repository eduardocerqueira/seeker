#date: 2024-11-18T17:13:26Z
#url: https://api.github.com/gists/10ebe4c5da1d511acd289ea5d5231e76
#owner: https://api.github.com/users/kaandesu

# !/bin/bash

cat >Makefile <<'EOF'
PROJECT_NAME = my_program
BUILD_DIR = build

all: clean build run

build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. > /dev/null
	@cd $(BUILD_DIR) && make > /dev/null

run: build
	@./$(BUILD_DIR)/$(PROJECT_NAME)

clean:
	@rm -rf $(BUILD_DIR)

EOF

echo "Makefile created."
