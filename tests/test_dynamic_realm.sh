#!/bin/bash
"""
Dynamic Realm Testing Script
Phase 4.5 Implementation - Convenient test execution wrapper
"""

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_RUNNER="$SCRIPT_DIR/run_dynamic_realm_tests.py"
PYTHON_CMD="python3"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
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

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Python
    if ! command -v $PYTHON_CMD &> /dev/null; then
        print_error "Python 3 not found. Please install Python 3."
        exit 1
    fi
    
    # Check test runner exists
    if [ ! -f "$TEST_RUNNER" ]; then
        print_error "Test runner not found: $TEST_RUNNER"
        exit 1
    fi
    
    # Check HTTP server availability
    if command -v curl &> /dev/null; then
        if curl -s -f "http://10.255.250.22:8080/mcp/health" > /dev/null 2>&1; then
            print_success "HTTP server is accessible"
        else
            print_warning "HTTP server not accessible - some tests may fail"
        fi
    else
        print_warning "curl not available - cannot check HTTP server"
    fi
    
    print_success "Prerequisites check completed"
}

# Function to show usage
show_usage() {
    echo "Dynamic Realm Testing Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h              Show this help message"
    echo "  --list-suites          List available test suites"
    echo "  --all                  Run all tests (including optional)"
    echo "  --unit                 Run unit tests only"
    echo "  --integration          Run integration tests only"
    echo "  --security             Run security tests only"
    echo "  --performance          Run performance tests only"
    echo "  --phase3               Run Phase 3 security tests only"
    echo "  --required             Run required tests only (default)"
    echo "  --verbose              Enable verbose output"
    echo "  --json-report          Generate JSON report only"
    echo ""
    echo "Examples:"
    echo "  $0                     # Run required tests"
    echo "  $0 --all               # Run all tests including performance"
    echo "  $0 --unit --security   # Run unit and security tests"
    echo "  $0 --verbose           # Run with detailed output"
    echo ""
}

# Function to run specific test suite
run_test_suite() {
    local suite_name="$1"
    print_status "Running $suite_name tests..."
    
    if $PYTHON_CMD "$TEST_RUNNER" --suites "$suite_name" "${VERBOSE_FLAG[@]}"; then
        print_success "$suite_name tests completed successfully"
        return 0
    else
        print_error "$suite_name tests failed"
        return 1
    fi
}

# Function to run all tests
run_all_tests() {
    print_status "Running all tests (including optional)..."
    
    if $PYTHON_CMD "$TEST_RUNNER" --include-optional "${VERBOSE_FLAG[@]}"; then
        print_success "All tests completed successfully"
        return 0
    else
        print_error "Some tests failed"
        return 1
    fi
}

# Function to run required tests only
run_required_tests() {
    print_status "Running required tests..."
    
    if $PYTHON_CMD "$TEST_RUNNER" "${VERBOSE_FLAG[@]}"; then
        print_success "Required tests completed successfully"
        return 0
    else
        print_error "Required tests failed"
        return 1
    fi
}

# Function to run multiple specific suites
run_multiple_suites() {
    local suites=("$@")
    print_status "Running test suites: ${suites[*]}"
    
    if $PYTHON_CMD "$TEST_RUNNER" --suites "${suites[@]}" "${VERBOSE_FLAG[@]}"; then
        print_success "Test suites completed successfully"
        return 0
    else
        print_error "Some test suites failed"
        return 1
    fi
}

# Function to list available test suites
list_suites() {
    print_status "Available test suites:"
    $PYTHON_CMD "$TEST_RUNNER" --list-suites
}

# Function to generate JSON report
generate_json_report() {
    print_status "Generating JSON test report..."
    
    # Run tests and generate report
    $PYTHON_CMD "$TEST_RUNNER" --include-optional "${VERBOSE_FLAG[@]}" 2>/dev/null || true
    
    # Check if report was generated
    if [ -f "$SCRIPT_DIR/test_report.json" ]; then
        print_success "JSON report generated: $SCRIPT_DIR/test_report.json"
        
        # Show summary from JSON report
        if command -v jq &> /dev/null; then
            echo ""
            echo "Test Summary:"
            jq -r '.summary | to_entries[] | "  \(.key): \(.value)"' "$SCRIPT_DIR/test_report.json"
        fi
    else
        print_warning "JSON report not found"
    fi
}

# Function to setup test environment
setup_test_environment() {
    print_status "Setting up test environment..."
    
    # Set Python path to include project directory
    export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
    
    # Create logs directory if it doesn't exist
    mkdir -p "$SCRIPT_DIR/logs"
    
    # Set timezone for consistent timestamps
    export TZ=UTC
    
    print_success "Test environment configured"
}

# Function to cleanup after tests
cleanup_test_environment() {
    print_status "Cleaning up test environment..."
    
    # Remove temporary test files
    find "$SCRIPT_DIR" -name "*.pyc" -delete 2>/dev/null || true
    find "$SCRIPT_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # Compress old logs if they exist
    if [ -d "$SCRIPT_DIR/logs" ] && [ "$(ls -A "$SCRIPT_DIR/logs")" ]; then
        find "$SCRIPT_DIR/logs" -name "*.log" -mtime +7 -exec gzip {} \; 2>/dev/null || true
    fi
    
    print_success "Cleanup completed"
}

# Main script logic
main() {
    local action="required"  # Default action
    local specific_suites=()
    local verbose_flag=()
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                show_usage
                exit 0
                ;;
            --list-suites)
                action="list"
                ;;
            --all)
                action="all"
                ;;
            --unit)
                specific_suites+=("unit")
                action="specific"
                ;;
            --integration)
                specific_suites+=("integration")
                action="specific"
                ;;
            --security)
                specific_suites+=("security")
                action="specific"
                ;;
            --performance)
                specific_suites+=("performance")
                action="specific"
                ;;
            --phase3)
                specific_suites+=("phase3")
                action="specific"
                ;;
            --required)
                action="required"
                ;;
            --verbose)
                verbose_flag=("--verbose")
                ;;
            --json-report)
                action="json-report"
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
        shift
    done
    
    # Set global verbose flag for functions
    VERBOSE_FLAG=("${verbose_flag[@]}")
    
    # Setup environment
    setup_test_environment
    
    # Check prerequisites
    check_prerequisites
    
    # Trap cleanup on exit
    trap cleanup_test_environment EXIT
    
    # Execute requested action
    case $action in
        "list")
            list_suites
            ;;
        "all")
            run_all_tests
            ;;
        "required")
            run_required_tests
            ;;
        "specific")
            if [ ${#specific_suites[@]} -eq 0 ]; then
                print_error "No specific test suites specified"
                exit 1
            fi
            run_multiple_suites "${specific_suites[@]}"
            ;;
        "json-report")
            generate_json_report
            ;;
        *)
            print_error "Unknown action: $action"
            exit 1
            ;;
    esac
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_success "Test execution completed successfully"
    else
        print_error "Test execution failed"
    fi
    
    exit $exit_code
}

# Run main function with all arguments
main "$@"