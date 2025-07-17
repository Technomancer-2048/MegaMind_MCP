# GitHub Issue: Critical JSON Parsing Bug in STDIO Bridge

## Issue Title
**🐛 Critical Bug: JSON Response Truncation in HardenedJSONParser Breaks MCP Function Calls**

## Labels
- `bug`
- `critical`
- `mcp-server`
- `stdio-bridge`
- `json-parsing`

## Priority
**HIGH** - Blocks all MCP function calls through Claude Code

## Description

### Summary
The MegaMind MCP STDIO bridge is failing to process HTTP responses from the backend server due to a line length limit in the `HardenedJSONParser._sanitize_input()` method. This causes all MCP function calls to fail with JSON parsing errors.

### Root Cause
The `HardenedJSONParser` in `mcp_server/json_utils.py` has a hardcoded `max_line_length = 10000` limit (line 214) that truncates HTTP responses at character 10000. Since the HTTP response containing all 19 MCP functions is 13,371 characters long and delivered as a single line, it gets truncated, breaking the JSON structure.

### Impact
- ❌ **All MCP function calls fail** with `JSONParsingError: JSON parsing failed after all recovery attempts`
- ❌ **Claude Code integration is broken** - cannot access any MegaMind functions
- ❌ **STDIO bridge returns backend errors** instead of successful function responses
- ✅ **HTTP endpoint works correctly** - only STDIO bridge affected

## Steps to Reproduce

1. **Test STDIO Bridge Function Call**:
   ```bash
   MEGAMIND_PROJECT_REALM=MegaMind_MCP MEGAMIND_PROJECT_NAME="MegaMind Context Database" MEGAMIND_DEFAULT_TARGET=PROJECT \
   echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | python3 mcp_server/stdio_http_bridge.py
   ```

2. **Expected Result**: List of 19 MCP functions
3. **Actual Result**: 
   ```json
   {"error":{"code":-32603,"message":"Backend error: JSON parsing failed in http_response after all recovery attempts"},"id":1,"jsonrpc":"2.0"}
   ```

## Diagnostic Information

### Error Details
```
ERROR: JSON parsing failed in http_response after all recovery attempts
DEBUG: Standard JSON parsing failed in test: Expecting ',' delimiter: line 1 column 10001 (char 10000)
```

### HTTP Response Analysis
- **Total Length**: 13,371 characters
- **Truncation Point**: Character 10,000
- **Character at 10,000**: `','` (comma)
- **Problem**: Truncation breaks JSON structure mid-object

### Function Exposure Verification
All 19 consolidated master functions are correctly exposed by the HTTP backend:
- ✅ Search Functions (3): `search_query`, `search_related`, `search_retrieve`
- ✅ Content Functions (4): `content_create`, `content_update`, `content_process`, `content_manage`
- ✅ Promotion Functions (3): `promotion_request`, `promotion_review`, `promotion_monitor`
- ✅ Session Functions (4): `session_create`, `session_manage`, `session_review`, `session_commit`
- ✅ AI Enhancement Functions (3): `ai_enhance`, `ai_learn`, `ai_analyze`
- ✅ Analytics Functions (2): `analytics_track`, `analytics_insights`

## Proposed Solution

### Immediate Fix
**File**: `mcp_server/json_utils.py`
**Line**: 214
**Change**: Increase `max_line_length` from 10,000 to at least 50,000

```python
# Current (broken)
max_line_length = 10000

# Proposed fix
max_line_length = 50000  # Support larger HTTP responses
```

### Alternative Solutions

1. **Adaptive Line Length**: Calculate required length based on payload size
2. **Disable Line Truncation**: Only truncate if security threat detected
3. **Multi-line Response**: Modify HTTP backend to format JSON with line breaks

### Code Location
- **File**: `/Data/MCP_Servers/MegaMind_MCP/mcp_server/json_utils.py`
- **Method**: `HardenedJSONParser._sanitize_input()`
- **Line**: 214

## Testing Verification

### Before Fix
```bash
# Test current behavior
curl -X POST http://10.255.250.22:8080 -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | wc -c
# Output: 13371 characters

# Test STDIO bridge (fails)
MEGAMIND_PROJECT_REALM=MegaMind_MCP MEGAMIND_PROJECT_NAME="MegaMind Context Database" MEGAMIND_DEFAULT_TARGET=PROJECT \
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | python3 mcp_server/stdio_http_bridge.py
# Output: JSON parsing error
```

### After Fix
```bash
# Test STDIO bridge (should work)
MEGAMIND_PROJECT_REALM=MegaMind_MCP MEGAMIND_PROJECT_NAME="MegaMind Context Database" MEGAMIND_DEFAULT_TARGET=PROJECT \
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | python3 mcp_server/stdio_http_bridge.py
# Expected: Valid JSON response with 19 functions
```

## Environment Information

- **MCP Server**: megamind-mcp-server-http (Docker container)
- **Bridge**: stdio_http_bridge.py (Python 3)
- **Backend**: HTTP endpoint at 10.255.250.22:8080
- **Response Size**: 13,371 characters
- **Current Limit**: 10,000 characters
- **Claude Code Integration**: .mcp.json configuration active

## Security Considerations

The line length limit was implemented for security (buffer overflow protection), but:
- ✅ **10KB limit is too restrictive** for legitimate MCP responses
- ✅ **50KB limit is reasonable** for MCP function lists
- ✅ **Payload size limit** (10MB) provides primary protection
- ✅ **JSON structure validation** provides secondary protection

## Urgency Justification

This bug **completely blocks Claude Code integration** with the MegaMind MCP server. All development workflows requiring MCP functions are non-functional until this is resolved.

## Related Issues

- This affects the Phase 5 Next-Generation AI function deployment
- Related to the STDIO-HTTP bridge architecture implemented in GitHub Issue #19
- Impacts the 19-function consolidation completed in the recent major refactoring

---

**Reporter**: Claude Code Investigation  
**Date**: 2025-07-17  
**Severity**: Critical  
**Component**: MCP Server - STDIO Bridge  
**Version**: Phase 5 AGI Ready