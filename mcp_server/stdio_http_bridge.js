#!/usr/bin/env node
/**
 * Node.js STDIO-to-HTTP Bridge for MegaMind MCP Server
 * High-performance bridge with comprehensive logging and security controls
 */

const readline = require('readline');
const axios = require('axios');
const { performance } = require('perf_hooks');

class STDIOHttpBridge {
    constructor(httpEndpoint = 'http://10.255.250.22:8080/mcp/jsonrpc') {
        this.httpEndpoint = httpEndpoint;
        this.healthEndpoint = 'http://10.255.250.22:8080/mcp/health';
        this.requestHeaders = {
            'Content-Type': 'application/json',
            'X-MCP-Realm-ID': 'MegaMind_MCP',
            'X-MCP-Project-Name': 'MegaMind_MCP'
        };
        
        // Setup readline interface for STDIO
        this.rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout,
            terminal: false
        });
        
        // Configure Axios with timeouts
        this.httpClient = axios.create({
            timeout: 30000, // 30 second timeout
            headers: this.requestHeaders
        });
        
        this.logLevel = process.env.LOG_LEVEL || 'INFO';
        this.setupLogging();
    }
    
    setupLogging() {
        // Log to stderr to avoid interfering with STDIO protocol
        this.log = {
            debug: (msg) => this.logLevel === 'DEBUG' && console.error(`[DEBUG] ${new Date().toISOString()} - ${msg}`),
            info: (msg) => ['DEBUG', 'INFO'].includes(this.logLevel) && console.error(`[INFO] ${new Date().toISOString()} - ${msg}`),
            warn: (msg) => console.error(`[WARN] ${new Date().toISOString()} - ${msg}`),
            error: (msg) => console.error(`[ERROR] ${new Date().toISOString()} - ${msg}`)
        };
    }
    
    /**
     * Sanitize request to block GLOBAL realm access and enforce PROJECT-only operations
     */
    sanitizeRequest(requestData) {
        const sanitized = JSON.parse(JSON.stringify(requestData)); // Deep clone
        
        // Check for realm_id in function arguments and block GLOBAL access
        if (sanitized.params && sanitized.params.arguments) {
            const args = sanitized.params.arguments;
            if (args.realm_id) {
                const realmId = args.realm_id;
                if (realmId === 'GLOBAL') {
                    this.log.warn(`Blocking GLOBAL realm access attempt in request ${requestData.id}`);
                    // Force to PROJECT realm instead
                    args.realm_id = 'PROJECT';
                } else if (!['PROJECT', 'MegaMind_MCP'].includes(realmId)) {
                    this.log.warn(`Blocking unauthorized realm '${realmId}' in request ${requestData.id}`);
                    // Force to PROJECT realm
                    args.realm_id = 'PROJECT';
                }
            }
        }
        
        return sanitized;
    }
    
    /**
     * Send HTTP request to backend with realm access controls
     */
    async sendHttpRequest(requestData) {
        const startTime = performance.now();
        
        try {
            const method = requestData.method || 'unknown';
            const requestId = requestData.id || 'unknown';
            
            this.log.info(`🔄 Processing request ${requestId}: ${method}`);
            this.log.debug(`📥 Raw request data: ${JSON.stringify(requestData).substring(0, 500)}...`);
            
            // Sanitize request to enforce PROJECT-only realm access
            const sanitizedRequest = this.sanitizeRequest(requestData);
            
            if (JSON.stringify(sanitizedRequest) !== JSON.stringify(requestData)) {
                this.log.warn(`🔒 Request sanitized for ${requestId}`);
                this.log.debug(`📝 Sanitized request: ${JSON.stringify(sanitizedRequest).substring(0, 500)}...`);
            }
            
            this.log.debug(`🌐 HTTP Request to ${this.httpEndpoint}`);
            this.log.debug(`📤 Headers: ${JSON.stringify(this.requestHeaders)}`);
            this.log.debug(`📤 Body size: ${JSON.stringify(sanitizedRequest).length} bytes`);
            this.log.debug(`📤 Body preview: ${JSON.stringify(sanitizedRequest).substring(0, 200)}...`);
            
            // Send request to HTTP backend
            const response = await this.httpClient.post(this.httpEndpoint, sanitizedRequest);
            
            const elapsed = performance.now() - startTime;
            this.log.info(`✅ Response for ${requestId}: HTTP ${response.status} (${elapsed.toFixed(3)}ms)`);
            this.log.debug(`📨 Response headers: ${JSON.stringify(response.headers)}`);
            this.log.debug(`📨 Response body: ${JSON.stringify(response.data).substring(0, 500)}...`);
            
            return response.data;
            
        } catch (error) {
            const elapsed = performance.now() - startTime;
            const requestId = requestData.id || 'unknown';
            
            if (error.response) {
                // HTTP error response
                this.log.error(`HTTP error for request ${requestId}: ${error.response.status} ${error.response.statusText}`);
                this.log.error(`Error body: ${JSON.stringify(error.response.data)}`);
                return {
                    jsonrpc: "2.0",
                    id: requestId,
                    error: {
                        code: -32603,
                        message: `HTTP ${error.response.status}: ${error.response.statusText}`
                    }
                };
            } else if (error.request) {
                // Network error
                this.log.error(`Network error for request ${requestId}: ${error.message}`);
                return {
                    jsonrpc: "2.0",
                    id: requestId,
                    error: {
                        code: -32603,
                        message: `Connection error: ${error.message}`
                    }
                };
            } else {
                // Other error
                this.log.error(`Unexpected error for request ${requestId}: ${error.message}`);
                return {
                    jsonrpc: "2.0",
                    id: requestId,
                    error: {
                        code: -32603,
                        message: `Bridge error: ${error.message}`
                    }
                };
            }
        }
    }
    
    /**
     * Handle MCP initialization requests - forward initialize to get actual capabilities
     */
    handleLocalMcpRequest(requestData) {
        const method = requestData.method || '';
        const requestId = requestData.id;
        
        if (method === 'initialize') {
            this.log.info("🤝 Forwarding MCP initialize request to HTTP backend for capabilities");
            // Forward to HTTP backend to get actual tool capabilities
            return null; // Signal to forward to HTTP backend
        } else if (method === 'notifications/initialized') {
            this.log.info("🎉 Client initialization complete - ready for normal operations");
            return 'no_response'; // Special signal for notifications
        } else {
            return null; // Not a local request, forward to HTTP backend
        }
    }
    
    /**
     * Test if HTTP backend is accessible
     */
    async testBackendConnectivity() {
        this.log.info("🔍 Testing HTTP backend connectivity...");
        try {
            this.log.debug(`🌐 Testing: ${this.healthEndpoint}`);
            
            const response = await this.httpClient.get(this.healthEndpoint);
            this.log.debug(`📨 Health response: ${JSON.stringify(response.data)}`);
            
            if (response.status === 200) {
                this.log.info("✅ HTTP backend is accessible and healthy");
                return true;
            } else {
                this.log.warn(`⚠️ HTTP backend returned status ${response.status}`);
                return false;
            }
        } catch (error) {
            this.log.error(`❌ Cannot reach HTTP backend: ${error.message}`);
            this.log.debug(`❌ Full error details: ${error.constructor.name}: ${error.message}`);
            return false;
        }
    }
    
    /**
     * Main STDIO loop - read from stdin, send to HTTP, write to stdout
     */
    async runStdioLoop() {
        this.log.info("🚀 Starting STDIO-HTTP bridge loop...");
        this.log.info(`🌐 HTTP Endpoint: ${this.httpEndpoint}`);
        this.log.info(`📤 Request Headers: ${JSON.stringify(this.requestHeaders)}`);
        
        return new Promise((resolve, reject) => {
            this.rl.on('line', async (line) => {
                try {
                    if (!line.trim()) {
                        this.log.debug("🔄 Empty line received, skipping");
                        return;
                    }
                    
                    this.log.debug(`📥 STDIN received: ${line.substring(0, 200)}...`);
                    
                    // Parse JSON-RPC request
                    const requestData = JSON.parse(line);
                    const requestId = requestData.id || 'unknown';
                    const method = requestData.method || 'unknown';
                    
                    this.log.info(`🔍 Parsed JSON-RPC: ID=${requestId}, Method=${method}`);
                    
                    // Check if this is a local MCP protocol request
                    const localResponse = this.handleLocalMcpRequest(requestData);
                    
                    if (localResponse === 'no_response') {
                        // Notification - no response needed
                        this.log.info(`✅ Processed notification ${method}`);
                    } else if (localResponse !== null) {
                        // Handle locally (MCP protocol requests)
                        const responseJson = JSON.stringify(localResponse);
                        this.log.debug(`📤 STDOUT sending (local): ${responseJson.substring(0, 200)}...`);
                        console.log(responseJson);
                        this.log.info(`✅ Completed local request ${requestId}`);
                    } else {
                        // Forward to HTTP backend for actual MCP tool calls
                        const responseData = await this.sendHttpRequest(requestData);
                        
                        // Write JSON-RPC response to stdout
                        const responseJson = JSON.stringify(responseData);
                        this.log.debug(`📤 STDOUT sending (HTTP): ${responseJson.substring(0, 200)}...`);
                        console.log(responseJson);
                        this.log.info(`✅ Completed HTTP request ${requestId}`);
                    }
                    
                } catch (error) {
                    if (error instanceof SyntaxError) {
                        // JSON parsing error
                        this.log.error(`❌ Invalid JSON received: ${error.message}`);
                        this.log.error(`❌ Raw line: ${line.substring(0, 100)}...`);
                        const errorResponse = {
                            jsonrpc: "2.0",
                            id: null,
                            error: {
                                code: -32700,
                                message: "Parse error: Invalid JSON"
                            }
                        };
                        const errorJson = JSON.stringify(errorResponse);
                        this.log.debug(`📤 STDOUT error: ${errorJson}`);
                        console.log(errorJson);
                    } else {
                        // Other error
                        this.log.error(`Error processing request: ${error.message}`);
                        const requestData = line ? (() => {
                            try { return JSON.parse(line); } catch { return {}; }
                        })() : {};
                        const errorResponse = {
                            jsonrpc: "2.0",
                            id: requestData.id || null,
                            error: {
                                code: -32603,
                                message: `Internal error: ${error.message}`
                            }
                        };
                        console.log(JSON.stringify(errorResponse));
                    }
                }
            });
            
            this.rl.on('close', () => {
                this.log.info("📜 EOF received on stdin, shutting down");
                resolve();
            });
            
            this.rl.on('error', (error) => {
                this.log.error(`STDIO error: ${error.message}`);
                reject(error);
            });
        });
    }
    
    /**
     * Main entry point
     */
    async run() {
        try {
            this.log.info("🎆 === MegaMind MCP STDIO-HTTP Bridge (Node.js) ===");
            this.log.info("🔗 Connecting Claude Code (STDIO) → HTTP MCP Server");
            this.log.info(`💻 Node.js version: ${process.version}`);
            this.log.info(`📁 Working directory: ${process.cwd()}`);
            
            // Test backend connectivity
            if (!(await this.testBackendConnectivity())) {
                this.log.error("❌ Backend connectivity test failed");
                process.exit(1);
            }
            
            // Initialize and run bridge
            this.log.info("🔄 Initializing STDIO-HTTP bridge...");
            this.log.info("✅ Bridge initialized successfully");
            
            await this.runStdioLoop();
            
        } catch (error) {
            this.log.error(`Bridge failed: ${error.message}`);
            this.log.debug(`Full error: ${error.stack}`);
            process.exit(1);
        }
    }
}

// Handle process signals
process.on('SIGINT', () => {
    console.error('[INFO] Received interrupt signal');
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.error('[INFO] Received terminate signal');
    process.exit(0);
});

// Main execution
if (require.main === module) {
    const bridge = new STDIOHttpBridge();
    bridge.run().catch((error) => {
        console.error(`[ERROR] Bridge startup failed: ${error.message}`);
        process.exit(1);
    });
}

module.exports = STDIOHttpBridge;