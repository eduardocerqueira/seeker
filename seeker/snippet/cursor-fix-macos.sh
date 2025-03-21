#date: 2025-03-21T17:00:56Z
#url: https://api.github.com/gists/b81ec008a75fcb85afe32142c65d294b
#owner: https://api.github.com/users/0PandaDEV

#!/bin/bash

# Set error handling
set -e

# Define log file path
LOG_FILE="/tmp/cursor_mac_id_modifier.log"

# Initialize log file
initialize_log() {
    echo "========== Cursor ID Modification Tool Log Start $(date) ==========" > "$LOG_FILE"
    chmod 644 "$LOG_FILE"
}

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log functions - output to both terminal and log file
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    echo "[WARN] $(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOG_FILE"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
    echo "[DEBUG] $(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOG_FILE"
}

# Log command output to log file
log_cmd_output() {
    local cmd="$1"
    local msg="$2"
    echo "[CMD] $(date '+%Y-%m-%d %H:%M:%S') Execute command: $cmd" >> "$LOG_FILE"
    echo "[CMD] $msg:" >> "$LOG_FILE"
    eval "$cmd" 2>&1 | tee -a "$LOG_FILE"
    echo "" >> "$LOG_FILE"
}

# Get current user
get_current_user() {
    if [ "$EUID" -eq 0 ]; then
        echo "$SUDO_USER"
    else
        echo "$USER"
    fi
}

CURRENT_USER=$(get_current_user)
if [ -z "$CURRENT_USER" ]; then
    log_error "Unable to get username"
    exit 1
fi

# Define configuration file paths
STORAGE_FILE="$HOME/Library/Application Support/Cursor/User/globalStorage/storage.json"
BACKUP_DIR="$HOME/Library/Application Support/Cursor/User/globalStorage/backups"

# Define Cursor application path
CURSOR_APP_PATH="/Applications/Cursor.app"

# Check permissions
check_permissions() {
    if [ "$EUID" -ne 0 ]; then
        log_error "Please run this script with sudo"
        echo "Example: sudo $0"
        exit 1
    fi
}

# Check and kill Cursor process
check_and_kill_cursor() {
    log_info "Checking Cursor process..."
    
    local attempt=1
    local max_attempts=5
    
    # Function: Get process details
    get_process_details() {
        local process_name="$1"
        log_debug "Getting $process_name process details:"
        ps aux | grep -i "/Applications/Cursor.app" | grep -v grep
    }
    
    while [ $attempt -le $max_attempts ]; do
        # Use more precise matching to get Cursor process
        CURSOR_PIDS=$(ps aux | grep -i "/Applications/Cursor.app" | grep -v grep | awk '{print $2}')
        
        if [ -z "$CURSOR_PIDS" ]; then
            log_info "No running Cursor process found"
            return 0
        fi
        
        log_warn "Cursor process is running"
        get_process_details "cursor"
        
        log_warn "Attempting to close Cursor process..."
        
        if [ $attempt -eq $max_attempts ]; then
            log_warn "Attempting to force terminate process..."
            kill -9 $CURSOR_PIDS 2>/dev/null || true
        else
            kill $CURSOR_PIDS 2>/dev/null || true
        fi
        
        sleep 1
        
        # Also use more precise matching to check if the process is still running
        if ! ps aux | grep -i "/Applications/Cursor.app" | grep -v grep > /dev/null; then
            log_info "Cursor process successfully closed"
            return 0
        fi
        
        log_warn "Waiting for process to close, attempt $attempt/$max_attempts..."
        ((attempt++))
    done
    
    log_error "Unable to close Cursor process after $max_attempts attempts"
    get_process_details "cursor"
    log_error "Please manually close the process and try again"
    exit 1
}

# Backup system ID
backup_system_id() {
    log_info "Backing up system ID..."
    local system_id_file="$BACKUP_DIR/system_id.backup_$(date +%Y%m%d_%H%M%S)"
    
    # Get and backup IOPlatformExpertDevice information
    {
        echo "# Original System ID Backup" > "$system_id_file"
        echo "## IOPlatformExpertDevice Info:" >> "$system_id_file"
        ioreg -rd1 -c IOPlatformExpertDevice >> "$system_id_file"
        
        chmod 444 "$system_id_file"
        chown "$CURRENT_USER" "$system_id_file"
        log_info "System ID backed up to: $system_id_file"
    } || {
        log_error "Failed to backup system ID"
        return 1
    }
}

# Backup configuration file
backup_config() {
    if [ ! -f "$STORAGE_FILE" ]; then
        log_warn "Configuration file does not exist, skipping backup"
        return 0
    fi
    
    mkdir -p "$BACKUP_DIR"
    local backup_file="$BACKUP_DIR/storage.json.backup_$(date +%Y%m%d_%H%M%S)"
    
    if cp "$STORAGE_FILE" "$backup_file"; then
        chmod 644 "$backup_file"
        chown "$CURRENT_USER" "$backup_file"
        log_info "Configuration backed up to: $backup_file"
    else
        log_error "Backup failed"
        exit 1
    fi
}

# Generate random ID
generate_random_id() {
    # Generate 32 bytes (64 hexadecimal characters) of random data
    openssl rand -hex 32
}

# Generate random UUID
generate_uuid() {
    uuidgen | tr '[:upper:]' '[:lower:]'
}

# Modify existing file
modify_or_add_config() {
    local key="$1"
    local value="$2"
    local file="$3"
    
    if [ ! -f "$file" ]; then
        log_error "File does not exist: $file"
        return 1
    fi
    
    # Ensure file is writable
    chmod 644 "$file" || {
        log_error "Unable to modify file permissions: $file"
        return 1
    }
    
    # Create temporary file
    local temp_file=$(mktemp)
    
    # Check if key exists
    if grep -q "\"$key\":" "$file"; then
        # Key exists, perform replacement
        sed "s/\"$key\":[[:space:]]*\"[^\"]*\"/\"$key\": \"$value\"/" "$file" > "$temp_file" || {
            log_error "Failed to modify configuration: $key"
            rm -f "$temp_file"
            return 1
        }
    else
        # Key doesn't exist, add new key-value pair
        sed "s/}$/,\n    \"$key\": \"$value\"\n}/" "$file" > "$temp_file" || {
            log_error "Failed to add configuration: $key"
            rm -f "$temp_file"
            return 1
        }
    fi
    
    # Check if temporary file is empty
    if [ ! -s "$temp_file" ]; then
        log_error "Generated temporary file is empty"
        rm -f "$temp_file"
        return 1
    fi
    
    # Use cat to replace original file content
    cat "$temp_file" > "$file" || {
        log_error "Unable to write to file: $file"
        rm -f "$temp_file"
        return 1
    }
    
    rm -f "$temp_file"
    
    # Restore file permissions
    chmod 444 "$file"
    
    return 0
}

# Generate new configuration
generate_new_config() {
  
    # Modify system ID
    log_info "Modifying system ID..."
    echo "[CONFIG] Starting system ID modification" >> "$LOG_FILE"
    
    # Backup current system ID
    backup_system_id
    
    # Generate new system UUID
    local new_system_uuid=$(uuidgen)
    echo "[CONFIG] Generated new system UUID: $new_system_uuid" >> "$LOG_FILE"
    
    # Modify system UUID
    sudo nvram SystemUUID="$new_system_uuid"
    echo "[CONFIG] System UUID has been set" >> "$LOG_FILE"
    
    printf "${YELLOW}System UUID has been updated to: $new_system_uuid${NC}\n"
    printf "${YELLOW}Please restart the system for changes to take effect${NC}\n"
    
    # Convert auth0|user_ to hexadecimal byte array
    local prefix_hex=$(echo -n "auth0|user_" | xxd -p)
    local random_part=$(generate_random_id)
    local machine_id="${prefix_hex}${random_part}"
    
    local mac_machine_id=$(generate_random_id)
    local device_id=$(generate_uuid | tr '[:upper:]' '[:lower:]')
    local sqm_id="{$(generate_uuid | tr '[:lower:]' '[:upper:]')}"
    
    echo "[CONFIG] Generated IDs:" >> "$LOG_FILE"
    echo "[CONFIG] machine_id: $machine_id" >> "$LOG_FILE"
    echo "[CONFIG] mac_machine_id: $mac_machine_id" >> "$LOG_FILE"
    echo "[CONFIG] device_id: $device_id" >> "$LOG_FILE"
    echo "[CONFIG] sqm_id: $sqm_id" >> "$LOG_FILE"
    
    log_info "Modifying configuration file..."
    # Check if configuration file exists
    if [ ! -f "$STORAGE_FILE" ]; then
        log_error "Configuration file not found: $STORAGE_FILE"
        log_warn "Please install and run Cursor once before using this script"
        exit 1
    fi
    
    # Ensure configuration file directory exists
    mkdir -p "$(dirname "$STORAGE_FILE")" || {
        log_error "Unable to create configuration directory"
        exit 1
    }
    
    # If file doesn't exist, create a basic JSON structure
    if [ ! -s "$STORAGE_FILE" ]; then
        echo '{}' > "$STORAGE_FILE" || {
            log_error "Unable to initialize configuration file"
            exit 1
        }
    fi
    
    # Modify existing file
    modify_or_add_config "telemetry.machineId" "$machine_id" "$STORAGE_FILE" || exit 1
    modify_or_add_config "telemetry.macMachineId" "$mac_machine_id" "$STORAGE_FILE" || exit 1
    modify_or_add_config "telemetry.devDeviceId" "$device_id" "$STORAGE_FILE" || exit 1
    modify_or_add_config "telemetry.sqmId" "$sqm_id" "$STORAGE_FILE" || exit 1
    
    # Set file permissions and owner
    chmod 444 "$STORAGE_FILE"  # Change to read-only permissions
    chown "$CURRENT_USER" "$STORAGE_FILE"
    
    # Verify permission settings
    if [ -w "$STORAGE_FILE" ]; then
        log_warn "Unable to set read-only permissions, trying other methods..."
        chattr +i "$STORAGE_FILE" 2>/dev/null || true
    else
        log_info "Successfully set file to read-only"
    fi
    
    echo
    log_info "Updated configuration: $STORAGE_FILE"
    log_debug "machineId: $machine_id"
    log_debug "macMachineId: $mac_machine_id"
    log_debug "devDeviceId: $device_id"
    log_debug "sqmId: $sqm_id"
}

# Clean up previous Cursor modifications
clean_cursor_app() {
    log_info "Attempting to clean up previous Cursor modifications..."
    
    # If backup exists, restore directly
    local latest_backup=""
    
    # Find the latest backup
    latest_backup=$(find /tmp -name "Cursor.app.backup_*" -type d -print 2>/dev/null | sort -r | head -1)
    
    if [ -n "$latest_backup" ] && [ -d "$latest_backup" ]; then
        log_info "Found existing backup: $latest_backup"
        log_info "Restoring original version..."
        
        # Stop Cursor process
        check_and_kill_cursor
        
        # Restore backup
        sudo rm -rf "$CURSOR_APP_PATH"
        sudo cp -R "$latest_backup" "$CURSOR_APP_PATH"
        sudo chown -R "$CURRENT_USER:staff" "$CURSOR_APP_PATH"
        sudo chmod -R 755 "$CURSOR_APP_PATH"
        
        log_info "Original version restored"
        return 0
    else
        log_warn "No existing backup found, attempting to reinstall Cursor..."
        echo "You can download and reinstall Cursor from https://cursor.sh"
        echo "Or continue with this script, which will attempt to fix the existing installation"
        
        # Logic for re-downloading and installing can be added here
        return 1
    fi
}

# Modify Cursor main program files (safe mode)
modify_cursor_app_files() {
    log_info "Safely modifying Cursor main program files..."
    log_info "Detailed logs will be recorded to: $LOG_FILE"
    
    # Clean up previous modifications first
    clean_cursor_app
    
    # Verify application exists
    if [ ! -d "$CURSOR_APP_PATH" ]; then
        log_error "Cursor.app not found, please confirm installation path: $CURSOR_APP_PATH"
        return 1
    fi

    # Define target files
    local target_files=(
        "${CURSOR_APP_PATH}/Contents/Resources/app/out/main.js"
        "${CURSOR_APP_PATH}/Contents/Resources/app/out/vs/code/node/cliProcessMain.js"
    )
    
    # Check if files exist and if they've been modified
    local need_modification=false
    local missing_files=false
    
    log_debug "Checking target files..."
    for file in "${target_files[@]}"; do
        if [ ! -f "$file" ]; then
            log_warn "File does not exist: ${file/$CURSOR_APP_PATH\//}"
            echo "[FILE_CHECK] File does not exist: $file" >> "$LOG_FILE"
            missing_files=true
            continue
        fi
        
        echo "[FILE_CHECK] File exists: $file ($(wc -c < "$file") bytes)" >> "$LOG_FILE"
        
        if ! grep -q "return crypto.randomUUID()" "$file" 2>/dev/null; then
            log_info "File needs modification: ${file/$CURSOR_APP_PATH\//}"
            grep -n "IOPlatformUUID" "$file" | head -3 >> "$LOG_FILE" || echo "[FILE_CHECK] IOPlatformUUID not found" >> "$LOG_FILE"
            need_modification=true
            break
        else
            log_info "File already modified: ${file/$CURSOR_APP_PATH\//}"
        fi
    done
    
    # If all files are already modified or don't exist, exit
    if [ "$missing_files" = true ]; then
        log_error "Some target files don't exist, please confirm Cursor installation is complete"
        return 1
    fi
    
    if [ "$need_modification" = false ]; then
        log_info "All target files have already been modified, no need to repeat"
        return 0
    fi

    # Create temporary working directory
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local temp_dir="/tmp/cursor_reset_${timestamp}"
    local temp_app="${temp_dir}/Cursor.app"
    local backup_app="/tmp/Cursor.app.backup_${timestamp}"
    
    log_debug "Creating temporary directory: $temp_dir"
    echo "[TEMP_DIR] Creating temporary directory: $temp_dir" >> "$LOG_FILE"
    
    # Clean up any existing old temporary directories
    if [ -d "$temp_dir" ]; then
        log_info "Cleaning up existing temporary directory..."
        rm -rf "$temp_dir"
    fi
    
    # Create new temporary directory
    mkdir -p "$temp_dir" || {
        log_error "Unable to create temporary directory: $temp_dir"
        echo "[ERROR] Unable to create temporary directory: $temp_dir" >> "$LOG_FILE"
        return 1
    }

    # Backup original application
    log_info "Backing up original application..."
    echo "[BACKUP] Starting backup: $CURSOR_APP_PATH -> $backup_app" >> "$LOG_FILE"
    
    cp -R "$CURSOR_APP_PATH" "$backup_app" || {
        log_error "Unable to create application backup"
        echo "[ERROR] Backup failed: $CURSOR_APP_PATH -> $backup_app" >> "$LOG_FILE"
        rm -rf "$temp_dir"
        return 1
    }
    
    echo "[BACKUP] Backup complete" >> "$LOG_FILE"

    # Copy application to temporary directory
    log_info "Creating temporary working copy..."
    echo "[COPY] Starting copy: $CURSOR_APP_PATH -> $temp_dir" >> "$LOG_FILE"
    
    cp -R "$CURSOR_APP_PATH" "$temp_dir" || {
        log_error "Unable to copy application to temporary directory"
        echo "[ERROR] Copy failed: $CURSOR_APP_PATH -> $temp_dir" >> "$LOG_FILE"
        rm -rf "$temp_dir" "$backup_app"
        return 1
    }
    
    echo "[COPY] Copy complete" >> "$LOG_FILE"

    # Ensure temporary directory has correct permissions
    chown -R "$CURRENT_USER:staff" "$temp_dir"
    chmod -R 755 "$temp_dir"

    # Remove signature (for better compatibility)
    log_info "Removing application signature..."
    echo "[CODESIGN] Removing signature: $temp_app" >> "$LOG_FILE"
    
    codesign --remove-signature "$temp_app" 2>> "$LOG_FILE" || {
        log_warn "Failed to remove application signature"
        echo "[WARN] Failed to remove signature: $temp_app" >> "$LOG_FILE"
    }

    # Remove signatures for all related components
    local components=(
        "$temp_app/Contents/Frameworks/Cursor Helper.app"
        "$temp_app/Contents/Frameworks/Cursor Helper (GPU).app"
        "$temp_app/Contents/Frameworks/Cursor Helper (Plugin).app"
        "$temp_app/Contents/Frameworks/Cursor Helper (Renderer).app"
    )

    for component in "${components[@]}"; do
        if [ -e "$component" ]; then
            log_info "Removing signature: $component"
            codesign --remove-signature "$component" || {
                log_warn "Failed to remove component signature: $component"
            }
        fi
    done
    
    # Modify target files
    local modified_count=0
    local files=(
        "${temp_app}/Contents/Resources/app/out/main.js"
        "${temp_app}/Contents/Resources/app/out/vs/code/node/cliProcessMain.js"
    )
    
    for file in "${files[@]}"; do
        if [ ! -f "$file" ]; then
            log_warn "File does not exist: ${file/$temp_dir\//}"
            continue
        fi
        
        log_debug "Processing file: ${file/$temp_dir\//}"
        echo "[PROCESS] Starting to process file: $file" >> "$LOG_FILE"
        echo "[PROCESS] File size: $(wc -c < "$file") bytes" >> "$LOG_FILE"
        
        # Output part of file content to log
        echo "[FILE_CONTENT] First 100 lines of file:" >> "$LOG_FILE"
        head -100 "$file" 2>/dev/null | grep -v "^$" | head -50 >> "$LOG_FILE"
        echo "[FILE_CONTENT] ..." >> "$LOG_FILE"
        
        # Create file backup
        cp "$file" "${file}.bak" || {
            log_error "Unable to create file backup: ${file/$temp_dir\//}"
            echo "[ERROR] Unable to create file backup: $file" >> "$LOG_FILE"
            continue
        }

        # Use sed for replacement instead of string operations
        if grep -q "IOPlatformUUID" "$file"; then
            log_debug "Found IOPlatformUUID keyword"
            echo "[FOUND] Found IOPlatformUUID keyword" >> "$LOG_FILE"
            grep -n "IOPlatformUUID" "$file" | head -5 >> "$LOG_FILE"
            
            # Locate IOPlatformUUID related function
            if grep -q "function a\$" "$file"; then
                # Check if already modified
                if grep -q "return crypto.randomUUID()" "$file"; then
                    log_info "File already contains randomUUID call, skipping modification"
                    ((modified_count++))
                    continue
                fi
                
                # Modify for code structure found in main.js
                if sed -i.tmp 's/function a\$(t){switch/function a\$(t){return crypto.randomUUID(); switch/' "$file"; then
                    log_debug "Successfully injected randomUUID call to a\$ function"
                    ((modified_count++))
                    log_info "Successfully modified file: ${file/$temp_dir\//}"
                else
                    log_error "Failed to modify a\$ function"
                    cp "${file}.bak" "$file"
                fi
            elif grep -q "async function v5" "$file"; then
                # Check if already modified
                if grep -q "return crypto.randomUUID()" "$file"; then
                    log_info "File already contains randomUUID call, skipping modification"
                    ((modified_count++))
                    continue
                fi
                
                # Alternative method - modify v5 function
                if sed -i.tmp 's/async function v5(t){let e=/async function v5(t){return crypto.randomUUID(); let e=/' "$file"; then
                    log_debug "Successfully injected randomUUID call to v5 function"
                    ((modified_count++))
                    log_info "Successfully modified file: ${file/$temp_dir\//}"
                else
                    log_error "Failed to modify v5 function"
                    cp "${file}.bak" "$file"
                fi
            else
                # Check if custom code has already been injected
                if grep -q "// Cursor ID Modification Tool Injection" "$file"; then
                    log_info "File already contains custom injected code, skipping modification"
                    ((modified_count++))
                    continue
                fi
                
                # Use more generic injection method
                log_warn "Specific function not found, trying generic modification method"
                inject_code="
// Cursor ID Modification Tool Injection - $(date +%Y%m%d%H%M%S)
// Random Device ID Generator Injection - $(date +%s)
const randomDeviceId_$(date +%s) = () => {
    try {
        return require('crypto').randomUUID();
    } catch (e) {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
            const r = Math.random() * 16 | 0;
            return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
        });
    }
};
"
                # Inject code at the beginning of the file
                echo "$inject_code" > "${file}.new"
                cat "$file" >> "${file}.new"
                mv "${file}.new" "$file"
                
                # Replace call points
                sed -i.tmp 's/await v5(!1)/randomDeviceId_'"$(date +%s)"'()/g' "$file"
                sed -i.tmp 's/a\$(t)/randomDeviceId_'"$(date +%s)"'()/g' "$file"
                
                log_debug "Generic modification complete"
                ((modified_count++))
                log_info "Successfully modified file using generic method: ${file/$temp_dir\//}"
            fi
        else
            # IOPlatformUUID not found, file structure might have changed
            log_warn "IOPlatformUUID not found, trying alternative methods"
            
            # Check if already injected or modified
            if grep -q "return crypto.randomUUID()" "$file" || grep -q "// Cursor ID Modification Tool Injection" "$file"; then
                log_info "File has already been modified, skipping"
                ((modified_count++))
                continue
            fi
            
            # Try to find other key functions like getMachineId or getDeviceId
            if grep -q "function t\$()" "$file" || grep -q "async function y5" "$file"; then
                log_debug "Found device ID related functions"
                
                # Modify MAC address retrieval function
                if grep -q "function t\$()" "$file"; then
                    sed -i.tmp 's/function t\$(){/function t\$(){return "00:00:00:00:00:00";/' "$file"
                    log_debug "Successfully modified MAC address retrieval function"
                fi
                
                # Modify device ID retrieval function
                if grep -q "async function y5" "$file"; then
                    sed -i.tmp 's/async function y5(t){/async function y5(t){return crypto.randomUUID();/' "$file"
                    log_debug "Successfully modified device ID retrieval function"
                fi
                
                ((modified_count++))
                log_info "Successfully modified file using alternative method: ${file/$temp_dir\//}"
            else
                # Last resort generic method - insert function definition overrides at the top of the file
                log_warn "No known functions found, using most generic method"
                
                inject_universal_code="
// Cursor ID Modification Tool Injection - $(date +%Y%m%d%H%M%S)
// Global Device Identifier Interception - $(date +%s)
const originalRequire_$(date +%s) = require;
require = function(module) {
    const result = originalRequire_$(date +%s)(module);
    if (module === 'crypto' && result.randomUUID) {
        const originalRandomUUID_$(date +%s) = result.randomUUID;
        result.randomUUID = function() {
            return '${new_uuid}';
        };
    }
    return result;
};

// Override all possible system ID retrieval functions
global.getMachineId = function() { return '${machine_id}'; };
global.getDeviceId = function() { return '${device_id}'; };
global.macMachineId = '${mac_machine_id}';
"
                # Inject code at the beginning of the file
                local new_uuid=$(uuidgen | tr '[:upper:]' '[:lower:]')
                local machine_id="auth0|user_$(openssl rand -hex 16)"
                local device_id=$(uuidgen | tr '[:upper:]' '[:lower:]')
                local mac_machine_id=$(openssl rand -hex 32)
                
                inject_universal_code=${inject_universal_code//\$\{new_uuid\}/$new_uuid}
                inject_universal_code=${inject_universal_code//\$\{machine_id\}/$machine_id}
                inject_universal_code=${inject_universal_code//\$\{device_id\}/$device_id}
                inject_universal_code=${inject_universal_code//\$\{mac_machine_id\}/$mac_machine_id}
                
                echo "$inject_universal_code" > "${file}.new"
                cat "$file" >> "${file}.new"
                mv "${file}.new" "$file"
                
                log_debug "Universal override complete"
                ((modified_count++))
                log_info "Successfully modified file using most generic method: ${file/$temp_dir\//}"
            fi
        fi
        
        # Add logging after key operations
        echo "[MODIFIED] File content after modification:" >> "$LOG_FILE"
        grep -n "return crypto.randomUUID()" "$file" | head -3 >> "$LOG_FILE"
        
        # Clean up temporary files
        rm -f "${file}.tmp" "${file}.bak"
        echo "[PROCESS] File processing completed: $file" >> "$LOG_FILE"
    done
    
    if [ "$modified_count" -eq 0 ]; then
        log_error "Failed to modify any files"
        rm -rf "$temp_dir"
        return 1
    fi
    
    # Re-sign application (with retry mechanism)
    local max_retry=3
    local retry_count=0
    local sign_success=false
    
    while [ $retry_count -lt $max_retry ]; do
        ((retry_count++))
        log_info "Attempting to sign (Attempt $retry_count)..."
        
        # Use more detailed signing parameters
        if codesign --sign - --force --deep --preserve-metadata=entitlements,identifier,flags "$temp_app" 2>&1 | tee /tmp/codesign.log; then
            # Verify signature
            if codesign --verify -vvvv "$temp_app" 2>/dev/null; then
                sign_success=true
                log_info "Application signature verification passed"
                break
            else
                log_warn "Signature verification failed, error log:"
                cat /tmp/codesign.log
            fi
        else
            log_warn "Signing failed, error log:"
            cat /tmp/codesign.log
        fi
        
        sleep 1
    done

    if ! $sign_success; then
        log_error "Failed to complete signing after $max_retry attempts"
        log_error "Please manually execute the following command to complete signing:"
        echo -e "${BLUE}sudo codesign --sign - --force --deep '${temp_app}'${NC}"
        echo -e "${YELLOW}After completion, please manually copy the application to the original path:${NC}"
        echo -e "${BLUE}sudo cp -R '${temp_app}' '/Applications/'${NC}"
        log_info "Temporary files retained at: ${temp_dir}"
        return 1
    fi

    # Replace original application
    log_info "Installing modified application..."
    if ! sudo rm -rf "$CURSOR_APP_PATH" || ! sudo cp -R "$temp_app" "/Applications/"; then
        log_error "Application replacement failed, restoring..."
        sudo rm -rf "$CURSOR_APP_PATH"
        sudo cp -R "$backup_app" "$CURSOR_APP_PATH"
        rm -rf "$temp_dir" "$backup_app"
        return 1
    fi
    
    # Clean up temporary files
    rm -rf "$temp_dir" "$backup_app"
    
    # Set permissions
    sudo chown -R "$CURRENT_USER:staff" "$CURSOR_APP_PATH"
    sudo chmod -R 755 "$CURSOR_APP_PATH"
    
    log_info "Cursor main program file modification complete! Original backup at: ${backup_app/$HOME/\~}"
    return 0
}

# Display file tree structure
show_file_tree() {
    local base_dir=$(dirname "$STORAGE_FILE")
    echo
    log_info "File structure:"
    echo -e "${BLUE}$base_dir${NC}"
    echo "├── globalStorage"
    echo "│   ├── storage.json (modified)"
    echo "│   └── backups"
    
    # List backup files
    if [ -d "$BACKUP_DIR" ]; then
        local backup_files=("$BACKUP_DIR"/*)
        if [ ${#backup_files[@]} -gt 0 ]; then
            for file in "${backup_files[@]}"; do
                if [ -f "$file" ]; then
                    echo "│       └── $(basename "$file")"
                fi
            done
        else
            echo "│       └── (empty)"
        fi
    fi
    echo
}

# Display follow information
show_follow_info() {
    echo
    echo -e "${GREEN}================================${NC}"
    echo -e "${YELLOW}  Follow WeChat Official Account [JianBingGuoZiJuanAI] to discuss more Cursor tips and AI knowledge (script is free, follow for more tips and experts in group) ${NC}"
    echo -e "${GREEN}================================${NC}"
    echo
}

# Disable auto update
disable_auto_update() {
    local updater_path="$HOME/Library/Application Support/Caches/cursor-updater"
    local app_update_yml="/Applications/Cursor.app/Contents/Resources/app-update.yml"
    
    echo
    log_info "Disabling Cursor auto-update..."
    
    # Backup and clear app-update.yml
    if [ -f "$app_update_yml" ]; then
        log_info "Backing up and modifying app-update.yml..."
        if ! sudo cp "$app_update_yml" "${app_update_yml}.bak" 2>/dev/null; then
            log_warn "Failed to backup app-update.yml, continuing..."
        fi
        
        if sudo bash -c "echo '' > \"$app_update_yml\"" && \
           sudo chmod 444 "$app_update_yml"; then
            log_info "Successfully disabled app-update.yml"
        else
            log_error "Failed to modify app-update.yml, please manually execute:"
            echo -e "${BLUE}sudo cp \"$app_update_yml\" \"${app_update_yml}.bak\"${NC}"
            echo -e "${BLUE}sudo bash -c 'echo \"\" > \"$app_update_yml\"'${NC}"
            echo -e "${BLUE}sudo chmod 444 \"$app_update_yml\"${NC}"
        fi
    else
        log_warn "app-update.yml file not found"
    fi
    
    # Also handle cursor-updater
    log_info "Processing cursor-updater..."
    if sudo rm -rf "$updater_path" && \
       sudo touch "$updater_path" && \
       sudo chmod 444 "$updater_path"; then
        log_info "Successfully disabled cursor-updater"
    else
        log_error "Failed to disable cursor-updater, please manually execute:"
        echo -e "${BLUE}sudo rm -rf \"$updater_path\" && sudo touch \"$updater_path\" && sudo chmod 444 \"$updater_path\"${NC}"
    fi
    
    echo
    log_info "Verification method:"
    echo "1. Run command: ls -l \"$updater_path\""
    echo "   Confirm file permissions show as: r--r--r--"
    echo "2. Run command: ls -l \"$app_update_yml\""
    echo "   Confirm file permissions show as: r--r--r--"
    echo
    log_info "Please restart Cursor after completion"
}

# Generate random MAC address
generate_random_mac() {
    # Generate random MAC address, keeping second bit of first byte as 0 (ensure unicast)
    printf '02:%02x:%02x:%02x:%02x:%02x' $((RANDOM%256)) $((RANDOM%256)) $((RANDOM%256)) $((RANDOM%256)) $((RANDOM%256))
}

# Get network interfaces list
get_network_interfaces() {
    networksetup -listallhardwareports | awk '/Hardware Port|Ethernet Address/ {print $NF}' | paste - - | grep -v 'N/A'
}

# Backup MAC addresses
backup_mac_addresses() {
    log_info "Backing up MAC addresses..."
    local backup_file="$BACKUP_DIR/mac_addresses.backup_$(date +%Y%m%d_%H%M%S)"
    
    {
        echo "# Original MAC Addresses Backup - $(date)" > "$backup_file"
        echo "## Network Interfaces:" >> "$backup_file"
        networksetup -listallhardwareports >> "$backup_file"
        
        chmod 444 "$backup_file"
        chown "$CURRENT_USER" "$backup_file"
        log_info "MAC addresses backed up to: $backup_file"
    } || {
        log_error "Failed to backup MAC addresses"
        return 1
    }
}

# Modify MAC address
modify_mac_address() {
    log_info "Getting network interface information..."
    
    # Backup current MAC addresses
    backup_mac_addresses
    
    # Get all network interfaces
    local interfaces=$(get_network_interfaces)
    
    if [ -z "$interfaces" ]; then
        log_error "No available network interfaces found"
        return 1
    fi
    
    echo
    log_info "Found following network interfaces:"
    echo "$interfaces" | nl -w2 -s') '
    echo
    
    echo -n "Please select interface number (press Enter to skip): "
    read -r choice
    
    if [ -z "$choice" ]; then
        log_info "Skipping MAC address modification"
        return 0
    fi
    
    # Get selected interface name
    local selected_interface=$(echo "$interfaces" | sed -n "${choice}p" | awk '{print $1}')
    
    if [ -z "$selected_interface" ]; then
        log_error "Invalid selection"
        return 1
    fi
    
    # Generate new MAC address
    local new_mac=$(generate_random_mac)
    
    log_info "Modifying MAC address for interface $selected_interface..."
    
    # Disable network interface
    sudo ifconfig "$selected_interface" down || {
        log_error "Unable to disable network interface"
        return 1
    }
    
    # Modify MAC address
    if sudo ifconfig "$selected_interface" ether "$new_mac"; then
        # Re-enable network interface
        sudo ifconfig "$selected_interface" up
        log_info "Successfully changed MAC address to: $new_mac"
        echo
        log_warn "Note: MAC address change may require reconnecting to network to take effect"
    else
        log_error "Failed to modify MAC address"
        # Try to restore network interface
        sudo ifconfig "$selected_interface" up
        return 1
    fi
}

# New restore feature option
restore_feature() {
    # Check if backup directory exists
    if [ ! -d "$BACKUP_DIR" ]; then
        log_warn "Backup directory does not exist"
        return 1
    fi

    # Use find command to get backup files list and store in array
    backup_files=()
    while IFS= read -r file; do
        [ -f "$file" ] && backup_files+=("$file")
    done < <(find "$BACKUP_DIR" -name "*.backup_*" -type f 2>/dev/null | sort)
    
    # Check if any backup files found
    if [ ${#backup_files[@]} -eq 0 ]; then
        log_warn "No backup files found"
        return 1
    fi
    
    echo
    log_info "Available backup files:"
    echo "0) Exit (default)"
    
    # Display backup files list
    for i in "${!backup_files[@]}"; do
        echo "$((i+1))) $(basename "${backup_files[$i]}")"
    done
    
    echo
    echo -n "Please select backup file number to restore [0-${#backup_files[@]}] (default: 0): "
    read -r choice
    
    # Handle user input
    if [ -z "$choice" ] || [ "$choice" = "0" ]; then
        log_info "Skipping restore operation"
        return 0
    fi
    
    # Validate input
    if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -gt "${#backup_files[@]}" ]; then
        log_error "Invalid selection"
        return 1
    fi
    
    # Get selected backup file
    local selected_backup="${backup_files[$((choice-1))]}"
    
    # Verify file existence and readability
    if [ ! -f "$selected_backup" ] || [ ! -r "$selected_backup" ]; then
        log_error "Cannot access selected backup file"
        return 1
    fi
    
    # Attempt to restore configuration
    if cp "$selected_backup" "$STORAGE_FILE"; then
        chmod 644 "$STORAGE_FILE"
        chown "$CURRENT_USER" "$STORAGE_FILE"
        log_info "Configuration restored from backup file: $(basename "$selected_backup")"
        return 0
    else
        log_error "Failed to restore configuration"
        return 1
    fi
}

# Fix "App is damaged and can't be opened" issue
fix_damaged_app() {
    log_info "Fixing 'App is damaged' issue..."
    
    # Check if Cursor app exists
    if [ ! -d "$CURSOR_APP_PATH" ]; then
        log_error "Cursor app not found: $CURSOR_APP_PATH"
        return 1
    fi
    
    log_info "Attempting to remove quarantine attribute..."
    if sudo xattr -rd com.apple.quarantine "$CURSOR_APP_PATH" 2>/dev/null; then
        log_info "Successfully removed quarantine attribute"
    else
        log_warn "Failed to remove quarantine attribute, trying other methods..."
    fi
    
    log_info "Attempting to re-sign application..."
    if sudo codesign --force --deep --sign - "$CURSOR_APP_PATH" 2>/dev/null; then
        log_info "Application re-signing successful"
    else
        log_warn "Application re-signing failed"
    fi
    
    echo
    log_info "Fix complete! Please try reopening Cursor application"
    echo
    echo -e "${YELLOW}If still unable to open, you can try these methods:${NC}"
    echo "1. In System Preferences -> Security & Privacy, click 'Open Anyway' button"
    echo "2. Temporarily disable Gatekeeper (not recommended): sudo spctl --master-disable"
    echo "3. Re-download and install Cursor application"
    echo
    echo -e "${BLUE}Reference link: https://sysin.org/blog/macos-if-crashes-when-opening/${NC}"
    
    return 0
}

# Main function
main() {
    
    # Initialize log file
    initialize_log
    log_info "Script started..."
    
    # Record system information
    log_info "System info: $(uname -a)"
    log_info "Current user: $CURRENT_USER"
    log_cmd_output "sw_vers" "macOS version info"
    log_cmd_output "which codesign" "codesign path"
    log_cmd_output "ls -la \"$CURSOR_APP_PATH\"" "Cursor app info"
    
    # Environment check
    if [[ $(uname) != "Darwin" ]]; then
        log_error "This script only supports macOS"
        exit 1
    fi
    
    clear
    # Display Logo
    echo -e "
    ██████╗██╗   ██╗██████╗ ███████╗ ██████╗ ██████╗ 
   ██╔════╝██║   ██║██╔══██╗██╔════╝██╔═══██╗██╔══██╗
   ██║     ██║   ██║██████╔╝███████╗██║   ██║██████╔╝
   ██║     ██║   ██║██╔══██╗╚════██║██║   ██║██╔══██╗
   ╚██████╗╚██████╔╝██║  ██║███████║╚██████╔╝██║  ██║
    ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝
    "
    echo -e "${BLUE}================================${NC}"
    echo -e "${GREEN}   Cursor Device ID Modification Tool ${NC}"
    echo -e "${YELLOW}  Follow WeChat: JianBingGuoZiJuanAI ${NC}"
    echo -e "${YELLOW}  Join us to discuss more Cursor tips and AI knowledge (Script is free, follow WeChat to join group for more tips) ${NC}"
    echo -e "${BLUE}================================${NC}"
    echo
    echo -e "${YELLOW}[Important Notice]${NC} This tool supports Cursor v0.47.x"
    echo -e "${YELLOW}[Important Notice]${NC} This tool is free, if it helps you, please follow WeChat: JianBingGuoZiJuanAI"
    echo
    
    # Execute main functions
    check_permissions
    check_and_kill_cursor
    backup_config
    generate_new_config
    
    # Ask user whether to modify main program files
    echo
    log_warn "Do you want to modify Cursor program files?"
    echo "0) No - Only modify config files (Safer but may need more frequent resets)"
    echo "1) Yes - Also modify program files (More persistent but small risk of instability)"
    echo ""
    printf "Enter choice [0-1] (default 1): "
    
    # Read user input robustly
    app_choice=""
    
    # Clear input buffer
    while read -r -t 0.1; do read -r; done 2>/dev/null
    
    exec <&-
    exec < /dev/tty
    
    app_choice=$(read -r choice; echo "$choice")
    if [ -z "$app_choice" ]; then
        if [ -e "/dev/tty" ] && [ -r "/dev/tty" ] && [ -w "/dev/tty" ]; then
            app_choice=$(head -n 1 < /dev/tty 2>/dev/null)
        fi
    fi
    
    echo "[INPUT_DEBUG] Choice read: '$app_choice'" >> "$LOG_FILE"
    
    set +e
    
    if [ "$app_choice" = "0" ]; then
        log_info "You chose to skip program file modification"
        log_info "Program file modification skipped"
    else
        log_info "Executing program file modification..."
        
        (
            if modify_cursor_app_files; then
                log_info "Program file modification successful!"
            else
                log_warn "Program file modification failed, but config changes may have succeeded"
                log_warn "If Cursor still shows device disabled after restart, please run this script again"
            fi
        )
    fi
    
    set -e
    
    # MAC address modification option
    echo
    log_warn "Do you want to modify MAC address?"
    echo "0) No - Keep default settings (default)"
    echo "1) Yes - Modify MAC address"
    echo ""
    printf "Enter choice [0-1] (default 0): "
    
    mac_choice=""
    
    while read -r -t 0.1; do read -r; done 2>/dev/null
    
    exec <&-
    exec < /dev/tty
    
    mac_choice=$(read -r choice; echo "$choice")
    if [ -z "$mac_choice" ]; then
        if [ -e "/dev/tty" ] && [ -r "/dev/tty" ] && [ -w "/dev/tty" ]; then
            mac_choice=$(head -n 1 < /dev/tty 2>/dev/null)
        fi
    fi
    
    echo "[INPUT_DEBUG] MAC choice: '$mac_choice'" >> "$LOG_FILE"
    
    set +e
    
    if [ "$mac_choice" = "1" ]; then
        log_info "You chose to modify MAC address"
        (
            if modify_mac_address; then
                log_info "MAC address modification complete!"
            else
                log_error "MAC address modification failed"
            fi
        )
    else
        log_info "MAC address modification skipped"
    fi
    
    set -e
    
    show_file_tree
    show_follow_info
  
    disable_auto_update

    log_info "Please restart Cursor to apply new configuration"

    show_follow_info

    # Repair options
    echo
    log_warn "Cursor Repair Options"
    echo "0) Skip - Don't perform repairs (default)"
    echo "1) Repair Mode - Restore original Cursor installation, fix errors from previous modifications"
    echo ""
    printf "Do you need to restore original Cursor? [0-1] (default 0): "
    
    fix_choice=""
    
    while read -r -t 0.1; do read -r; done 2>/dev/null
    
    exec <&-
    exec < /dev/tty
    
    fix_choice=$(read -r choice; echo "$choice")
    if [ -z "$fix_choice" ]; then
        if [ -e "/dev/tty" ] && [ -r "/dev/tty" ] && [ -w "/dev/tty" ]; then
            fix_choice=$(head -n 1 < /dev/tty 2>/dev/null)
        fi
    fi
    
    echo "[INPUT_DEBUG] Repair choice: '$fix_choice'" >> "$LOG_FILE"
    
    set +e
    
    if [ "$fix_choice" = "1" ]; then
        log_info "You chose repair mode"
        (
            if clean_cursor_app; then
                log_info "Cursor restored to original state"
                log_info "If you need ID modification, please run this script again"
            else
                log_warn "No backup found, cannot auto-restore"
                log_warn "Recommend reinstalling Cursor"
            fi
        )
    else
        log_info "Repair operation skipped"
    fi
    
    set -e

    log_info "Script execution complete"
    echo "========== Cursor ID Modification Tool Log End $(date) ==========" >> "$LOG_FILE"
    
    echo
    log_info "Detailed log saved to: $LOG_FILE"
    echo "If you encounter issues, please provide this log file to developers for troubleshooting"
    echo
    
    # "App is damaged" repair option
    echo
    log_warn "App Repair Options"
    echo "0) Skip - Don't perform repairs (default)"
    echo "1) Fix 'App is damaged' issue - Resolve macOS warning about damaged app"
    echo ""
    printf "Do you need to fix 'App is damaged' issue? [0-1] (default 0): "
    
    damaged_choice=""
    while read -r -t 0.1; do read -r; done 2>/dev/null
    exec <&-
    exec < /dev/tty
    damaged_choice=$(read -r choice; echo "$choice")
    if [ -z "$damaged_choice" ]; then
        if [ -e "/dev/tty" ] && [ -r "/dev/tty" ] && [ -w "/dev/tty" ]; then
            damaged_choice=$(head -n 1 < /dev/tty 2>/dev/null)
        fi
    fi
    
    echo "[INPUT_DEBUG] App repair choice: '$damaged_choice'" >> "$LOG_FILE"
    
    set +e
    
    if [ "$damaged_choice" = "1" ]; then
        log_info "You chose to fix 'App is damaged' issue"
        (
            if fix_damaged_app; then
                log_info "Fixed 'App is damaged' issue"
            else
                log_warn "Failed to fix 'App is damaged' issue"
            fi
        )
    else
        log_info "App repair operation skipped"
    fi
    
    set -e
}

# Execute main function
main
