#date: 2025-09-01T17:12:35Z
#url: https://api.github.com/gists/b32c2620f2aa644d71f58aa74f07d6cb
#owner: https://api.github.com/users/vrahmanov

#!/bin/bash

# Kubernetes Deployment Troubleshooter
# This script can be sourced from anywhere and will automatically detect deployment issues
# Usage: source k8s-deployment-troubleshooter.sh && troubleshoot_deployment <namespace>

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}[$(date '+%Y-%m-%d %H:%M:%S')] ${message}${NC}"
}

# Function to check if kubectl is available
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        print_status $RED "ERROR: kubectl is not installed or not in PATH"
        return 1
    fi
    
    # Check if we can connect to cluster
    if ! kubectl cluster-info &> /dev/null; then
        print_status $RED "ERROR: Cannot connect to Kubernetes cluster"
        return 1
    fi
    
    print_status $GREEN "✓ kubectl is available and connected to cluster"
}

# Function to check namespace exists
check_namespace() {
    local namespace=$1
    
    if ! kubectl get namespace "$namespace" &> /dev/null; then
        print_status $RED "ERROR: Namespace '$namespace' does not exist"
        return 1
    fi
    
    print_status $GREEN "✓ Namespace '$namespace' exists"
}

# Function to get deployment status
get_deployment_status() {
    local namespace=$1
    local deployments
    
    deployments=$(kubectl get deployments -n "$namespace" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -z "$deployments" ]]; then
        print_status $YELLOW "No deployments found in namespace '$namespace'"
        return 0
    fi
    
    print_status $BLUE "Found deployments: $deployments"
    
    for deployment in $deployments; do
        print_status $BLUE "\n--- Analyzing deployment: $deployment ---"
        analyze_deployment "$namespace" "$deployment"
    done
}

# Function to analyze a specific deployment
analyze_deployment() {
    local namespace=$1
    local deployment=$2
    
    # Get deployment status
    local status=$(kubectl get deployment "$deployment" -n "$namespace" -o jsonpath='{.status}' 2>/dev/null)
    
    if [[ -z "$status" ]]; then
        print_status $RED "Cannot get status for deployment '$deployment'"
        return 1
    fi
    
    # Check if deployment is available
    local available=$(echo "$status" | jq -r '.availableReplicas // 0' 2>/dev/null || echo "0")
    local desired=$(echo "$status" | jq -r '.replicas // 0' 2>/dev/null || echo "0")
    local updated=$(echo "$status" | jq -r '.updatedReplicas // 0' 2>/dev/null || echo "0")
    local ready=$(echo "$status" | jq -r '.readyReplicas // 0' 2>/dev/null || echo "0")
    
    print_status $BLUE "Desired replicas: $desired"
    print_status $BLUE "Available replicas: $available"
    print_status $BLUE "Updated replicas: $updated"
    print_status $BLUE "Ready replicas: $ready"
    
    if [[ "$available" -eq "$desired" && "$available" -gt 0 ]]; then
        print_status $GREEN "✓ Deployment '$deployment' is healthy"
        return 0
    else
        print_status $RED "✗ Deployment '$deployment' has issues"
        investigate_deployment_issues "$namespace" "$deployment"
    fi
}

# Function to investigate deployment issues
investigate_deployment_issues() {
    local namespace=$1
    local deployment=$2
    
    print_status $YELLOW "Investigating deployment issues..."
    
    # Check deployment events
    print_status $BLUE "Checking deployment events..."
    kubectl get events -n "$namespace" --field-selector involvedObject.name="$deployment" --sort-by='.lastTimestamp' | head -20
    
    # Check replica set
    print_status $BLUE "Checking replica sets..."
    local rs_name=$(kubectl get rs -n "$namespace" -l "app.kubernetes.io/name=$deployment" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -n "$rs_name" ]]; then
        print_status $BLUE "Replica set: $rs_name"
        kubectl describe rs "$rs_name" -n "$namespace" | grep -A 20 "Events:" || true
    fi
    
    # Check pods
    print_status $BLUE "Checking pods..."
    local pods=$(kubectl get pods -n "$namespace" -l "app.kubernetes.io/name=$deployment" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -n "$pods" ]]; then
        for pod in $pods; do
            print_status $BLUE "\n--- Pod: $pod ---"
            analyze_pod "$namespace" "$pod"
        done
    else
        print_status $YELLOW "No pods found for deployment '$deployment'"
    fi
}

# Function to analyze a specific pod
analyze_pod() {
    local namespace=$1
    local pod=$2
    
    # Get pod status
    local phase=$(kubectl get pod "$pod" -n "$namespace" -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown")
    local ready=$(kubectl get pod "$pod" -n "$namespace" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "Unknown")
    
    print_status $BLUE "Pod phase: $phase"
    print_status $BLUE "Ready status: $ready"
    
    if [[ "$phase" == "Running" && "$ready" == "True" ]]; then
        print_status $GREEN "✓ Pod '$pod' is healthy"
        return 0
    else
        print_status $RED "✗ Pod '$pod' has issues"
        investigate_pod_issues "$namespace" "$pod"
    fi
}

# Function to investigate pod issues
investigate_pod_issues() {
    local namespace=$1
    local pod=$2
    
    print_status $YELLOW "Investigating pod issues..."
    
    # Check pod events
    print_status $BLUE "Checking pod events..."
    kubectl get events -n "$namespace" --field-selector involvedObject.name="$pod" --sort-by='.lastTimestamp' | head -10
    
    # Check pod description
    print_status $BLUE "Pod description:"
    kubectl describe pod "$pod" -n "$namespace" | grep -E "(Events:|Status:|Conditions:|Containers:|Image:|Ports:|Mounts:|Volumes:)" -A 20 || true
    
    # Check container logs if pod is running
    local phase=$(kubectl get pod "$pod" -n "$namespace" -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown")
    if [[ "$phase" == "Running" ]]; then
        print_status $BLUE "Container logs (last 20 lines):"
        kubectl logs "$pod" -n "$namespace" --tail=20 || true
    fi
    
    # Check container statuses
    print_status $BLUE "Container statuses:"
    kubectl get pod "$pod" -n "$namespace" -o jsonpath='{.status.containerStatuses[*].name}' | tr ' ' '\n' | while read -r container; do
        if [[ -n "$container" ]]; then
            local container_ready=$(kubectl get pod "$pod" -n "$namespace" -o jsonpath="{.status.containerStatuses[?(@.name=='$container')].ready}")
            local container_restart_count=$(kubectl get pod "$pod" -n "$namespace" -o jsonpath="{.status.containerStatuses[?(@.name=='$container')].restartCount}")
            local container_state=$(kubectl get pod "$pod" -n "$namespace" -o jsonpath="{.status.containerStatuses[?(@.name=='$container')].state}")
            
            print_status $BLUE "  Container: $container"
            print_status $BLUE "    Ready: $container_ready"
            print_status $BLUE "    Restart count: $container_restart_count"
            print_status $BLUE "    State: $container_state"
            
            # Check if container is waiting
            local waiting_reason=$(kubectl get pod "$pod" -n "$namespace" -o jsonpath="{.status.containerStatuses[?(@.name=='$container')].state.waiting.reason}" 2>/dev/null || echo "")
            if [[ -n "$waiting_reason" ]]; then
                print_status $YELLOW "    Waiting reason: $waiting_reason"
            fi
        fi
    done
}

# Function to check service and ingress
check_networking() {
    local namespace=$1
    
    print_status $BLUE "\n--- Checking networking ---"
    
    # Check services
    local services=$(kubectl get services -n "$namespace" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")
    if [[ -n "$services" ]]; then
        print_status $BLUE "Services found: $services"
        for service in $services; do
            local service_type=$(kubectl get service "$service" -n "$namespace" -o jsonpath='{.spec.type}' 2>/dev/null || echo "Unknown")
            local cluster_ip=$(kubectl get service "$service" -n "$namespace" -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "None")
            print_status $BLUE "  $service ($service_type) - ClusterIP: $cluster_ip"
        done
    else
        print_status $YELLOW "No services found"
    fi
    
    # Check ingress
    local ingress=$(kubectl get ingress -n "$namespace" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")
    if [[ -n "$ingress" ]]; then
        print_status $BLUE "Ingress found: $ingress"
        for ing in $ingress; do
            local ingress_class=$(kubectl get ingress "$ing" -n "$namespace" -o jsonpath='{.spec.ingressClassName}' 2>/dev/null || echo "Default")
            local hosts=$(kubectl get ingress "$ing" -n "$namespace" -o jsonpath='{.spec.rules[*].host}' 2>/dev/null || echo "None")
            print_status $BLUE "  $ing (Class: $ingress_class) - Hosts: $hosts"
        done
    else
        print_status $YELLOW "No ingress found"
    fi
}

# Function to check persistent volumes
check_storage() {
    local namespace=$1
    
    print_status $BLUE "\n--- Checking storage ---"
    
    local pvcs=$(kubectl get pvc -n "$namespace" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")
    if [[ -n "$pvcs" ]]; then
        print_status $BLUE "PVCs found: $pvcs"
        for pvc in $pvcs; do
            local status=$(kubectl get pvc "$pvc" -n "$namespace" -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown")
            local capacity=$(kubectl get pvc "$pvc" -n "$namespace" -o jsonpath='{.status.capacity.storage}' 2>/dev/null || echo "Unknown")
            print_status $BLUE "  $pvc - Status: $status, Capacity: $capacity"
            
            if [[ "$status" != "Bound" ]]; then
                print_status $YELLOW "    PVC '$pvc' is not bound"
                kubectl describe pvc "$pvc" -n "$namespace" | grep -A 10 "Events:" || true
            fi
        done
    else
        print_status $YELLOW "No PVCs found"
    fi
}

# Function to check configmaps and secrets
check_configuration() {
    local namespace=$1
    
    print_status $BLUE "\n--- Checking configuration ---"
    
    # Check configmaps
    local configmaps=$(kubectl get configmaps -n "$namespace" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")
    if [[ -n "$configmaps" ]]; then
        print_status $BLUE "ConfigMaps found: $configmaps"
    else
        print_status $YELLOW "No ConfigMaps found"
    fi
    
    # Check secrets
    local secrets= "**********"='{.items[*].metadata.name}' 2>/dev/null || echo "")
    if [[ -n "$secrets" ]]; then
        print_status $BLUE "Secrets found: "**********"
    else
        print_status $YELLOW "No Secrets found"
    fi
}

# Main troubleshooting function
troubleshoot_deployment() {
    local namespace=$1
    
    if [[ -z "$namespace" ]]; then
        print_status $RED "ERROR: Please provide a namespace name"
        print_status $YELLOW "Usage: troubleshoot_deployment <namespace>"
        return 1
    fi
    
    print_status $BLUE "=========================================="
    print_status $BLUE "Kubernetes Deployment Troubleshooter"
    print_status $BLUE "Namespace: $namespace"
    print_status $BLUE "=========================================="
    
    # Check prerequisites
    check_kubectl || return 1
    check_namespace "$namespace" || return 1
    
    # Start troubleshooting
    print_status $BLUE "\nStarting comprehensive deployment analysis..."
    
    # Check deployments
    get_deployment_status "$namespace"
    
    # Check networking
    check_networking "$namespace"
    
    # Check storage
    check_storage "$namespace"
    
    # Check configuration
    check_configuration "$namespace"
    
    print_status $BLUE "\n=========================================="
    print_status $BLUE "Troubleshooting complete for namespace: $namespace"
    print_status $BLUE "=========================================="
}

# Function to show usage
show_usage() {
    cat << EOF
Kubernetes Deployment Troubleshooter

This script can be sourced from anywhere and will automatically detect deployment issues.

Usage:
    source k8s-deployment-troubleshooter.sh
    troubleshoot_deployment <namespace>

Examples:
    source k8s-deployment-troubleshooter.sh
    troubleshoot_deployment default
    
    source k8s-deployment-troubleshooter.sh
    troubleshoot_deployment my-app

Features:
    - Automatic deployment health detection
    - Pod status analysis
    - Container log inspection
    - Event monitoring
    - Networking verification
    - Storage validation
    - Configuration checking
    - Comprehensive error reporting

Requirements:
    - kubectl installed and configured
    - Access to Kubernetes cluster
    - Valid namespace name

EOF
}

# Check if script is being sourced
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    print_status $GREEN "Kubernetes Deployment Troubleshooter loaded successfully!"
    print_status $BLUE "Use 'troubleshoot_deployment <namespace>' to start troubleshooting"
    print_status $BLUE "Use 'show_usage' to see detailed usage information"
else
    print_status $YELLOW "This script is designed to be sourced, not executed directly."
    print_status $YELLOW "Please run: source $(basename "$0")"
    exit 1
fi

$(basename "$0")"
    exit 1
fi

