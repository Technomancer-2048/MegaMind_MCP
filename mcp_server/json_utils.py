#!/usr/bin/env python3
"""
Hardened JSON Parsing Utilities for MegaMind MCP Server
Provides robust, secure JSON parsing with comprehensive error handling and validation
"""

import json
import logging
import re
import sys
from typing import Any, Dict, List, Optional, Union, Tuple
from decimal import Decimal, InvalidOperation
from datetime import datetime
import unicodedata

logger = logging.getLogger(__name__)

class JSONParsingError(Exception):
    """Custom exception for JSON parsing errors"""
    def __init__(self, message: str, error_code: str = None, original_error: Exception = None):
        super().__init__(message)
        self.error_code = error_code
        self.original_error = original_error

class JSONValidationError(Exception):
    """Custom exception for JSON validation errors"""
    def __init__(self, message: str, path: str = None):
        super().__init__(message)
        self.path = path

class HardenedJSONParser:
    """
    Hardened JSON parser with security and robustness features:
    - Size limits and payload validation
    - Unicode normalization and sanitization
    - Structure validation and depth limits
    - Error recovery and detailed error reporting
    - Memory-efficient parsing for large payloads
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Size limits (configurable)
        self.max_payload_size = self.config.get('max_payload_size', 10 * 1024 * 1024)  # 10MB
        self.max_string_length = self.config.get('max_string_length', 1024 * 1024)     # 1MB
        self.max_depth = self.config.get('max_depth', 100)
        self.max_keys = self.config.get('max_keys', 10000)
        
        # Parsing options
        self.strict_mode = self.config.get('strict_mode', True)
        self.normalize_unicode = self.config.get('normalize_unicode', True)
        self.sanitize_strings = self.config.get('sanitize_strings', True)
        self.validate_mcp_structure = self.config.get('validate_mcp_structure', True)
        
        # Dangerous patterns to detect
        self.dangerous_patterns = [
            re.compile(r'__[a-zA-Z_]+__'),  # Python dunder methods
            re.compile(r'eval\s*\('),       # eval calls
            re.compile(r'exec\s*\('),       # exec calls
            re.compile(r'import\s+'),       # import statements
            re.compile(r'subprocess'),      # subprocess calls
            re.compile(r'os\.system'),      # system calls
        ]
        
        logger.debug(f"HardenedJSONParser initialized: max_size={self.max_payload_size}, strict={self.strict_mode}")
    
    def safe_loads(self, json_string: str, context: str = "unknown") -> Dict[str, Any]:
        """
        Safely parse JSON string with comprehensive validation and error handling
        
        Args:
            json_string: Raw JSON string to parse
            context: Context description for better error reporting
            
        Returns:
            Parsed JSON object as dictionary
            
        Raises:
            JSONParsingError: For various parsing and validation failures
        """
        try:
            # Pre-validation checks
            self._validate_input_size(json_string, context)
            
            # Normalize and sanitize input
            if self.normalize_unicode:
                json_string = self._normalize_unicode(json_string)
            
            if self.sanitize_strings:
                json_string = self._sanitize_input(json_string)
            
            # Parse JSON with error recovery
            parsed_data = self._parse_with_recovery(json_string, context)
            
            # Post-parsing validation
            self._validate_structure(parsed_data, context)
            
            # MCP-specific validation
            if self.validate_mcp_structure:
                self._validate_mcp_protocol(parsed_data, context)
            
            # Security scanning
            self._scan_for_security_issues(parsed_data, context)
            
            logger.debug(f"Successfully parsed JSON for context: {context}")
            return parsed_data
            
        except JSONParsingError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error parsing JSON for {context}: {e}")
            raise JSONParsingError(
                f"Unexpected parsing error in {context}: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                original_error=e
            )
    
    def safe_dumps(self, obj: Any, context: str = "unknown", **kwargs) -> str:
        """
        Safely serialize object to JSON with size and content validation
        
        Args:
            obj: Object to serialize
            context: Context for error reporting
            **kwargs: Additional arguments for json.dumps
            
        Returns:
            JSON string representation
            
        Raises:
            JSONParsingError: If serialization fails or output is invalid
        """
        try:
            # Set safe defaults
            safe_kwargs = {
                'ensure_ascii': kwargs.get('ensure_ascii', False),
                'separators': kwargs.get('separators', (',', ':')),
                'sort_keys': kwargs.get('sort_keys', True),
                'cls': kwargs.get('cls', MegaMindJSONEncoder),
            }
            safe_kwargs.update(kwargs)
            
            # Validate object before serialization
            self._validate_object_for_serialization(obj, context)
            
            # Serialize
            json_string = json.dumps(obj, **safe_kwargs)
            
            # Validate output size
            if len(json_string.encode('utf-8')) > self.max_payload_size:
                raise JSONParsingError(
                    f"Serialized JSON exceeds size limit in {context}",
                    error_code="OUTPUT_TOO_LARGE"
                )
            
            return json_string
            
        except json.JSONDecodeError as e:
            raise JSONParsingError(
                f"JSON serialization failed in {context}: {str(e)}",
                error_code="SERIALIZATION_ERROR",
                original_error=e
            )
        except Exception as e:
            logger.error(f"Unexpected error serializing JSON for {context}: {e}")
            raise JSONParsingError(
                f"Unexpected serialization error in {context}: {str(e)}",
                error_code="UNEXPECTED_SERIALIZATION_ERROR",
                original_error=e
            )
    
    def _validate_input_size(self, json_string: str, context: str):
        """Validate input size limits"""
        byte_size = len(json_string.encode('utf-8'))
        if byte_size > self.max_payload_size:
            raise JSONParsingError(
                f"JSON payload too large in {context}: {byte_size} bytes > {self.max_payload_size}",
                error_code="PAYLOAD_TOO_LARGE"
            )
        
        if byte_size == 0:
            raise JSONParsingError(
                f"Empty JSON payload in {context}",
                error_code="EMPTY_PAYLOAD"
            )
    
    def _normalize_unicode(self, json_string: str) -> str:
        """Normalize Unicode characters to prevent encoding attacks"""
        try:
            # Normalize to NFC form (canonical decomposition followed by canonical composition)
            normalized = unicodedata.normalize('NFC', json_string)
            
            # Remove or replace problematic Unicode characters
            # Keep only printable ASCII, basic Unicode, and essential JSON characters
            cleaned = ''.join(
                char for char in normalized
                if unicodedata.category(char) not in ['Cc', 'Cf', 'Cs', 'Co', 'Cn'] or char in ['\n', '\r', '\t']
            )
            
            return cleaned
            
        except Exception as e:
            logger.warning(f"Unicode normalization failed: {e}")
            return json_string  # Fall back to original if normalization fails
    
    def _sanitize_input(self, json_string: str) -> str:
        """Sanitize input to remove potentially dangerous content"""
        # Remove null bytes
        sanitized = json_string.replace('\x00', '')
        
        # Limit line length to prevent buffer overflow attacks
        lines = sanitized.split('\n')
        max_line_length = 10000
        sanitized_lines = [
            line[:max_line_length] if len(line) > max_line_length else line
            for line in lines
        ]
        sanitized = '\n'.join(sanitized_lines)
        
        return sanitized
    
    def _parse_with_recovery(self, json_string: str, context: str) -> Dict[str, Any]:
        """Parse JSON with error recovery strategies"""
        
        # Strategy 1: Standard parsing
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            logger.debug(f"Standard JSON parsing failed in {context}: {e}")
        
        # Strategy 2: Try to fix common issues
        if not self.strict_mode:
            try:
                fixed_json = self._attempt_json_repair(json_string)
                if fixed_json != json_string:
                    logger.info(f"Attempting JSON repair in {context}")
                    return json.loads(fixed_json)
            except json.JSONDecodeError:
                logger.debug(f"JSON repair failed in {context}")
        
        # Strategy 3: Try to extract valid JSON subset
        if not self.strict_mode:
            try:
                subset = self._extract_json_subset(json_string)
                if subset:
                    logger.warning(f"Using partial JSON extraction in {context}")
                    return json.loads(subset)
            except json.JSONDecodeError:
                logger.debug(f"JSON subset extraction failed in {context}")
        
        # All strategies failed
        raise JSONParsingError(
            f"JSON parsing failed in {context} after all recovery attempts",
            error_code="PARSE_ERROR"
        )
    
    def _attempt_json_repair(self, json_string: str) -> str:
        """Attempt to repair common JSON formatting issues"""
        repaired = json_string.strip()
        
        # Fix common trailing comma issues
        repaired = re.sub(r',\s*}', '}', repaired)
        repaired = re.sub(r',\s*]', ']', repaired)
        
        # Fix unquoted keys (basic attempt)
        repaired = re.sub(r'(\w+):', r'"\1":', repaired)
        
        # Fix single quotes to double quotes
        repaired = re.sub(r"'([^']*)'", r'"\1"', repaired)
        
        return repaired
    
    def _extract_json_subset(self, json_string: str) -> Optional[str]:
        """Extract the largest valid JSON subset from malformed input"""
        # Try to find complete JSON objects
        brace_count = 0
        for i, char in enumerate(json_string):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found complete object
                    candidate = json_string[:i+1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        continue
        
        return None
    
    def _validate_structure(self, data: Any, context: str, current_depth: int = 0):
        """Validate JSON structure for security and resource limits"""
        
        if current_depth > self.max_depth:
            raise JSONValidationError(
                f"JSON depth exceeds limit ({self.max_depth}) in {context}",
                path=f"depth:{current_depth}"
            )
        
        if isinstance(data, dict):
            if len(data) > self.max_keys:
                raise JSONValidationError(
                    f"Too many keys ({len(data)} > {self.max_keys}) in {context}",
                    path="root"
                )
            
            for key, value in data.items():
                # Validate key
                if not isinstance(key, str):
                    raise JSONValidationError(
                        f"Non-string key found in {context}: {type(key)}",
                        path=str(key)
                    )
                
                if len(key) > 1000:  # Reasonable key length limit
                    raise JSONValidationError(
                        f"Key too long in {context}: {len(key)} characters",
                        path=key[:100]
                    )
                
                # Recursively validate value
                self._validate_structure(value, context, current_depth + 1)
        
        elif isinstance(data, list):
            if len(data) > self.max_keys:  # Use same limit for array elements
                raise JSONValidationError(
                    f"Too many array elements ({len(data)} > {self.max_keys}) in {context}",
                    path="array"
                )
            
            for i, item in enumerate(data):
                self._validate_structure(item, context, current_depth + 1)
        
        elif isinstance(data, str):
            if len(data) > self.max_string_length:
                raise JSONValidationError(
                    f"String too long in {context}: {len(data)} > {self.max_string_length}",
                    path="string"
                )
    
    def _validate_mcp_protocol(self, data: Dict[str, Any], context: str):
        """Validate MCP protocol structure"""
        if not isinstance(data, dict):
            raise JSONValidationError(
                f"MCP message must be an object in {context}",
                path="root"
            )
        
        # Check required MCP fields
        if 'jsonrpc' not in data:
            raise JSONValidationError(
                f"Missing 'jsonrpc' field in MCP message in {context}",
                path="jsonrpc"
            )
        
        if data.get('jsonrpc') != '2.0':
            raise JSONValidationError(
                f"Invalid JSON-RPC version in {context}: {data.get('jsonrpc')}",
                path="jsonrpc"
            )
        
        # Validate ID field if present
        if 'id' in data:
            id_value = data['id']
            if id_value is not None and not isinstance(id_value, (str, int, float)):
                raise JSONValidationError(
                    f"Invalid ID type in MCP message in {context}: {type(id_value)}",
                    path="id"
                )
        
        # Validate method field for requests
        if 'method' in data:
            method = data['method']
            if not isinstance(method, str):
                raise JSONValidationError(
                    f"Method must be string in {context}: {type(method)}",
                    path="method"
                )
            
            if len(method) > 100:  # Reasonable method name limit
                raise JSONValidationError(
                    f"Method name too long in {context}: {len(method)}",
                    path="method"
                )
    
    def _scan_for_security_issues(self, data: Any, context: str, path: str = ""):
        """Scan for potential security issues in parsed data"""
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                
                # Check for dangerous key names
                if any(pattern.search(key) for pattern in self.dangerous_patterns):
                    logger.warning(f"Potentially dangerous key detected in {context}: {key}")
                
                self._scan_for_security_issues(value, context, current_path)
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]" if path else f"[{i}]"
                self._scan_for_security_issues(item, context, current_path)
        
        elif isinstance(data, str):
            # Check for dangerous string patterns
            for pattern in self.dangerous_patterns:
                if pattern.search(data):
                    logger.warning(f"Potentially dangerous string content in {context} at {path}")
                    break
    
    def _validate_object_for_serialization(self, obj: Any, context: str):
        """Validate object before serialization"""
        # Check for circular references (basic check)
        try:
            # This will raise ValueError for circular references
            json.dumps(obj, cls=MegaMindJSONEncoder)
        except ValueError as e:
            if "circular reference" in str(e).lower():
                raise JSONParsingError(
                    f"Circular reference detected in {context}",
                    error_code="CIRCULAR_REFERENCE",
                    original_error=e
                )
            raise

class MegaMindJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for MegaMind-specific types"""
    
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            # For custom objects, serialize their __dict__
            return obj.__dict__
        return super().default(obj)

# Global parser instance with default configuration
_default_parser = HardenedJSONParser({
    'max_payload_size': 10 * 1024 * 1024,  # 10MB
    'max_string_length': 1024 * 1024,      # 1MB
    'max_depth': 100,
    'max_keys': 10000,
    'strict_mode': False,  # Allow repair attempts
    'normalize_unicode': True,
    'sanitize_strings': True,
    'validate_mcp_structure': True,
})

# Convenience functions for backward compatibility
def safe_json_loads(json_string: str, context: str = "unknown") -> Dict[str, Any]:
    """Safely parse JSON string using default hardened parser"""
    return _default_parser.safe_loads(json_string, context)

def safe_json_dumps(obj: Any, context: str = "unknown", **kwargs) -> str:
    """Safely serialize object to JSON using default hardened parser"""
    return _default_parser.safe_dumps(obj, context, **kwargs)

def create_parser(config: Dict[str, Any]) -> HardenedJSONParser:
    """Create a custom hardened JSON parser with specific configuration"""
    return HardenedJSONParser(config)

# Utility functions for common validation tasks
def validate_mcp_request(data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate MCP request structure
    Returns: (is_valid, error_message)
    """
    try:
        _default_parser._validate_mcp_protocol(data, "mcp_request")
        return True, ""
    except JSONValidationError as e:
        return False, str(e)

def sanitize_for_logging(obj: Any, max_length: int = 500) -> str:
    """Safely serialize object for logging with size limits"""
    try:
        json_str = safe_json_dumps(obj, "logging")
        if len(json_str) > max_length:
            return json_str[:max_length] + "... [truncated]"
        return json_str
    except Exception as e:
        return f"<unserializable: {type(obj).__name__}> Error: {str(e)}"

def clean_decimal_objects(obj: Any) -> Any:
    """Recursively convert Decimal objects to float for JSON serialization"""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: clean_decimal_objects(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_decimal_objects(item) for item in obj]
    return obj

if __name__ == "__main__":
    # Test the hardened parser
    test_cases = [
        '{"test": "valid"}',
        '{"test": "valid",}',  # trailing comma
        "{'test': 'single quotes'}",  # single quotes
        '{"test": "unicode: 🚀"}',
        '{"深度": "nested", "test": {"inner": "value"}}',  # Unicode keys
    ]
    
    parser = HardenedJSONParser()
    for i, test_case in enumerate(test_cases):
        try:
            result = parser.safe_loads(test_case, f"test_{i}")
            print(f"✅ Test {i}: {result}")
        except Exception as e:
            print(f"❌ Test {i}: {e}")