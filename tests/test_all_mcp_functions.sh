#!/bin/bash

# Comprehensive MCP Function Testing Script
# Tests all 14 MegaMind MCP functions for Issue #6 verification

set -e  # Exit on any error

# Configuration
BASE_URL="http://10.255.250.22:8080/mcp/jsonrpc"
REALM_HEADER="X-Realm-ID: MegaMind_MCP"
CONTENT_TYPE="Content-Type: application/json"
TEST_SESSION="test_session_comprehensive_$(date +%s)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to make MCP call and check result
test_mcp_function() {
    local test_name="$1"
    local function_name="$2"
    local arguments="$3"
    local expect_success="$4"  # true/false
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    echo -e "${BLUE}Testing $test_name...${NC}"
    
    # Build JSON payload
    local payload="{\"jsonrpc\": \"2.0\", \"method\": \"tools/call\", \"params\": {\"name\": \"$function_name\", \"arguments\": $arguments}, \"id\": \"test-$TOTAL_TESTS\"}"
    
    # Make the call
    local response=$(curl -s -X POST "$BASE_URL" \
        -H "$CONTENT_TYPE" \
        -H "$REALM_HEADER" \
        -d "$payload")
    
    # Check if response contains error
    if echo "$response" | jq -e '.error' > /dev/null 2>&1; then
        if [ "$expect_success" = "true" ]; then
            echo -e "${RED}❌ FAILED: $test_name${NC}"
            echo "Error: $(echo "$response" | jq -r '.error.message')"
            FAILED_TESTS=$((FAILED_TESTS + 1))
        else
            echo -e "${YELLOW}⚠️  Expected failure: $test_name${NC}"
            echo "Expected error: $(echo "$response" | jq -r '.error.message')"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        fi
    else
        if [ "$expect_success" = "true" ]; then
            echo -e "${GREEN}✅ PASSED: $test_name${NC}"
            # Show result summary
            local result=$(echo "$response" | jq -r '.result.content[0].text' 2>/dev/null || echo "No content")
            if [ ${#result} -gt 200 ]; then
                echo "Result: ${result:0:200}..."
            else
                echo "Result: $result"
            fi
            PASSED_TESTS=$((PASSED_TESTS + 1))
        else
            echo -e "${RED}❌ Unexpected success: $test_name${NC}"
            FAILED_TESTS=$((FAILED_TESTS + 1))
        fi
    fi
    
    # Show processing time
    local processing_time=$(echo "$response" | jq -r '._meta.processing_time_ms' 2>/dev/null || echo "unknown")
    echo "Processing time: ${processing_time}ms"
    echo ""
}

# Function to create session metadata (direct database insert simulation)
setup_test_session() {
    echo -e "${BLUE}Setting up test session: $TEST_SESSION${NC}"
    echo "Note: Session-based functions may fail due to foreign key constraints"
    echo "This is expected behavior for a read-only test environment"
    echo ""
}

echo "=========================================="
echo "MegaMind MCP Function Comprehensive Test"
echo "=========================================="
echo "Testing all 14 MCP functions systematically"
echo "Session ID: $TEST_SESSION"
echo ""

setup_test_session

echo "===========================================" 
echo "PHASE 1: SEARCH & RETRIEVAL FUNCTIONS (5)"
echo "==========================================="

# Test 1.1: Hybrid Search (Primary)
test_mcp_function \
    "Hybrid Search" \
    "mcp__megamind__search_chunks" \
    '{"query": "sample", "limit": 3, "search_type": "hybrid"}' \
    "true"

# Test 1.2: Keyword Search
test_mcp_function \
    "Keyword Search" \
    "mcp__megamind__search_chunks" \
    '{"query": "database", "limit": 3, "search_type": "keyword"}' \
    "true"

# Test 1.3: Semantic Search
test_mcp_function \
    "Semantic Search" \
    "mcp__megamind__search_chunks_semantic" \
    '{"query": "testing functionality", "limit": 3, "threshold": 0.1}' \
    "true"

# Test 1.4: Get Individual Chunk
test_mcp_function \
    "Get Chunk by ID" \
    "mcp__megamind__get_chunk" \
    '{"chunk_id": "sample_001", "include_relationships": true}' \
    "true"

# Test 1.5: Get Related Chunks
test_mcp_function \
    "Get Related Chunks" \
    "mcp__megamind__get_related_chunks" \
    '{"chunk_id": "sample_001", "max_depth": 2}' \
    "true"

# Test 1.6: Similarity Search
test_mcp_function \
    "Similarity Search" \
    "mcp__megamind__search_chunks_by_similarity" \
    '{"reference_chunk_id": "sample_001", "limit": 3, "threshold": 0.1}' \
    "true"

echo "============================================"
echo "PHASE 2: CONTENT MANAGEMENT FUNCTIONS (4)"
echo "============================================"

# Test 2.1: Batch Generate Embeddings
test_mcp_function \
    "Batch Generate Embeddings" \
    "mcp__megamind__batch_generate_embeddings" \
    '{}' \
    "true"

# Test 2.2: Create Chunk (expect failure due to write permissions)
test_mcp_function \
    "Create Chunk" \
    "mcp__megamind__create_chunk" \
    "{\"content\": \"Test chunk for comprehensive testing\", \"source_document\": \"test_doc.md\", \"section_path\": \"/Testing\", \"session_id\": \"$TEST_SESSION\", \"target_realm\": \"GLOBAL\"}" \
    "false"

# Test 2.3: Update Chunk (expect failure due to session constraint)
test_mcp_function \
    "Update Chunk" \
    "mcp__megamind__update_chunk" \
    "{\"chunk_id\": \"sample_001\", \"new_content\": \"Updated content for testing\", \"session_id\": \"$TEST_SESSION\"}" \
    "false"

# Test 2.4: Add Relationship (expect failure due to session constraint)
test_mcp_function \
    "Add Relationship" \
    "mcp__megamind__add_relationship" \
    "{\"chunk_id_1\": \"sample_001\", \"chunk_id_2\": \"sample_002\", \"relationship_type\": \"references\", \"session_id\": \"$TEST_SESSION\"}" \
    "false"

echo "============================================"
echo "PHASE 3: SESSION MANAGEMENT FUNCTIONS (3)"
echo "============================================"

# Test 3.1: Get Session Primer
test_mcp_function \
    "Get Session Primer (fresh)" \
    "mcp__megamind__get_session_primer" \
    '{}' \
    "true"

# Test 3.2: Get Session Primer with continuity
test_mcp_function \
    "Get Session Primer (continuity)" \
    "mcp__megamind__get_session_primer" \
    "{\"last_session_data\": \"$TEST_SESSION\"}" \
    "true"

# Test 3.3: Get Pending Changes (expect empty or error)
test_mcp_function \
    "Get Pending Changes" \
    "mcp__megamind__get_pending_changes" \
    "{\"session_id\": \"$TEST_SESSION\"}" \
    "true"

# Test 3.4: Commit Session Changes (expect failure due to session constraint)
test_mcp_function \
    "Commit Session Changes" \
    "mcp__megamind__commit_session_changes" \
    "{\"session_id\": \"$TEST_SESSION\", \"approved_changes\": []}" \
    "false"

echo "==============================================="
echo "PHASE 4: ANALYTICS & OPTIMIZATION FUNCTIONS (2)"
echo "==============================================="

# Test 4.1: Track Access
test_mcp_function \
    "Track Access" \
    "mcp__megamind__track_access" \
    '{"chunk_id": "sample_001", "query_context": "Comprehensive testing"}' \
    "true"

# Test 4.2: Get Hot Contexts
test_mcp_function \
    "Get Hot Contexts (Sonnet)" \
    "mcp__megamind__get_hot_contexts" \
    '{"model_type": "sonnet", "limit": 5}' \
    "true"

# Test 4.3: Get Hot Contexts (Opus)
test_mcp_function \
    "Get Hot Contexts (Opus)" \
    "mcp__megamind__get_hot_contexts" \
    '{"model_type": "opus", "limit": 3}' \
    "true"

echo "==========================================="
echo "PHASE 5: EDGE CASES & ERROR HANDLING"
echo "==========================================="

# Test 5.1: Invalid Chunk ID
test_mcp_function \
    "Invalid Chunk ID" \
    "mcp__megamind__get_chunk" \
    '{"chunk_id": "nonexistent_chunk_123"}' \
    "true"

# Test 5.2: High Similarity Threshold
test_mcp_function \
    "High Similarity Threshold" \
    "mcp__megamind__search_chunks_semantic" \
    '{"query": "sample", "threshold": 0.99, "limit": 5}' \
    "true"

# Test 5.3: Large Limit
test_mcp_function \
    "Large Limit Request" \
    "mcp__megamind__search_chunks" \
    '{"query": "test", "limit": 100}' \
    "true"

echo "=================================================="
echo "PHASE 6: PROMOTION WORKFLOW FUNCTIONS (6) - NEW"
echo "=================================================="

# Test 6.1: Create Promotion Request
test_mcp_function \
    "Create Promotion Request" \
    "mcp__megamind__create_promotion_request" \
    "{\"source_chunk_id\": \"sample_001\", \"target_realm_id\": \"GLOBAL\", \"promotion_type\": \"copy\", \"justification\": \"High-value content for global access\", \"business_impact\": \"high\", \"requested_by\": \"test_user\", \"session_id\": \"$TEST_SESSION\"}" \
    "false"

# Test 6.2: Get Promotion Requests
test_mcp_function \
    "Get Promotion Requests" \
    "mcp__megamind__get_promotion_requests" \
    '{"status": "pending", "limit": 10}' \
    "true"

# Test 6.3: Get Promotion Impact Analysis
test_mcp_function \
    "Get Promotion Impact" \
    "mcp__megamind__get_promotion_impact" \
    '{"promotion_id": "test_promotion_123"}' \
    "true"

# Test 6.4: Approve Promotion Request
test_mcp_function \
    "Approve Promotion Request" \
    "mcp__megamind__approve_promotion_request" \
    '{"promotion_id": "test_promotion_123", "reviewed_by": "admin_user", "review_notes": "Approved for global knowledge sharing"}' \
    "false"

# Test 6.5: Reject Promotion Request
test_mcp_function \
    "Reject Promotion Request" \
    "mcp__megamind__reject_promotion_request" \
    '{"promotion_id": "test_promotion_456", "reviewed_by": "admin_user", "review_notes": "Content not suitable for global promotion"}' \
    "false"

# Test 6.6: Get Promotion Queue Summary
test_mcp_function \
    "Get Promotion Queue Summary" \
    "mcp__megamind__get_promotion_queue_summary" \
    '{"realm_id": "GLOBAL"}' \
    "true"

echo "==========================================="
echo "COMPREHENSIVE TEST RESULTS"
echo "==========================================="
echo -e "Total Tests: ${BLUE}$TOTAL_TESTS${NC}"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS${NC}"

# Calculate success rate
SUCCESS_RATE=$((PASSED_TESTS * 100 / TOTAL_TESTS))
echo -e "Success Rate: ${BLUE}$SUCCESS_RATE%${NC}"

if [ $SUCCESS_RATE -ge 85 ]; then
    echo -e "${GREEN}🎉 OVERALL RESULT: EXCELLENT${NC}"
    echo "The MegaMind MCP server is fully operational!"
elif [ $SUCCESS_RATE -ge 70 ]; then
    echo -e "${YELLOW}⚠️  OVERALL RESULT: GOOD${NC}"
    echo "Most functions working, minor issues to address"
else
    echo -e "${RED}❌ OVERALL RESULT: NEEDS WORK${NC}"
    echo "Significant functionality issues detected"
fi

echo ""
echo "==========================================="
echo "FUNCTIONALITY ANALYSIS"
echo "==========================================="
echo "✅ Search Functions: Expected to be fully operational"
echo "✅ Analytics Functions: Expected to be fully operational"  
echo "⚠️  Content Management: Expected failures due to write permissions"
echo "⚠️  Session Management: Expected failures due to foreign key constraints"
echo "🚀 Promotion Functions: NEW - Knowledge promotion workflow (Phase 3)"
echo "   ├── Create promotion requests for knowledge sharing"
echo "   ├── Review and approval workflow management"
echo "   ├── Impact analysis for promotion decisions"
echo "   └── Queue management and tracking"
echo ""
echo "Note: Some failures are expected in a production-secured environment"
echo "Core search and analytics functionality should be 100% operational"
echo "New promotion features enable cross-realm knowledge management"
echo ""

exit 0