#!/usr/bin/env node
/**
 * Test script for Node.js STDIO-HTTP Bridge
 */

const { spawn } = require('child_process');
const path = require('path');

async function testBridge() {
    console.log('🧪 Testing Node.js STDIO-HTTP Bridge...');
    
    const bridgePath = path.join(__dirname, 'stdio_http_bridge.js');
    const bridge = spawn('node', [bridgePath], {
        stdio: ['pipe', 'pipe', 'pipe']
    });
    
    let output = '';
    let errors = '';
    
    bridge.stdout.on('data', (data) => {
        output += data.toString();
    });
    
    bridge.stderr.on('data', (data) => {
        errors += data.toString();
    });
    
    // Test 1: Initialize
    console.log('📋 Test 1: MCP Initialize');
    const initRequest = {
        jsonrpc: "2.0",
        id: 1,
        method: "initialize",
        params: {
            protocolVersion: "2024-11-05",
            capabilities: { roots: { listChanged: true } },
            clientInfo: { name: "test-client", version: "1.0.0" }
        }
    };
    
    bridge.stdin.write(JSON.stringify(initRequest) + '\n');
    
    // Wait for response
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Test 2: Function call
    console.log('📋 Test 2: Function Call');
    const funcRequest = {
        jsonrpc: "2.0",
        id: 2,
        method: "tools/call",
        params: {
            name: "mcp__megamind__search_chunks",
            arguments: { query: "test", limit: 1 }
        }
    };
    
    bridge.stdin.write(JSON.stringify(funcRequest) + '\n');
    
    // Wait for response
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Close bridge
    bridge.stdin.end();
    
    await new Promise((resolve) => {
        bridge.on('close', (code) => {
            console.log(`\n🔍 Bridge exited with code: ${code}`);
            
            // Parse responses
            const responses = output.trim().split('\n').filter(line => line.trim());
            console.log(`📊 Received ${responses.length} responses`);
            
            if (responses.length > 0) {
                try {
                    const initResponse = JSON.parse(responses[0]);
                    if (initResponse.result && initResponse.result.capabilities) {
                        const toolCount = Object.keys(initResponse.result.capabilities.tools || {}).length;
                        console.log(`✅ Initialize: Found ${toolCount} MCP functions`);
                    } else {
                        console.log(`❌ Initialize: Invalid response format`);
                    }
                } catch (e) {
                    console.log(`❌ Initialize: Parse error - ${e.message}`);
                }
                
                if (responses.length > 1) {
                    try {
                        const funcResponse = JSON.parse(responses[1]);
                        if (funcResponse.result && funcResponse.result.content) {
                            console.log(`✅ Function Call: Success`);
                        } else if (funcResponse.error) {
                            console.log(`❌ Function Call: Error - ${funcResponse.error.message}`);
                        } else {
                            console.log(`❌ Function Call: Unexpected response format`);
                        }
                    } catch (e) {
                        console.log(`❌ Function Call: Parse error - ${e.message}`);
                    }
                } else {
                    console.log(`⚠️  Function Call: No response received`);
                }
            } else {
                console.log(`❌ No responses received`);
            }
            
            if (errors) {
                console.log('\n📜 Bridge Logs:');
                console.log(errors.split('\n').slice(-10).join('\n')); // Last 10 lines
            }
            
            resolve();
        });
    });
}

testBridge().catch(console.error);