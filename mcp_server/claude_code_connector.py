#!/usr/bin/env python3
"""
Claude Code Connector - Enhanced STDIO bridge with connection resilience
"""

import sys
import os
import json
import time
import subprocess
import signal
import logging
from pathlib import Path

# Configure logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)

def main():
    logger.info("=== Claude Code Enhanced Connector ===")
    logger.info("Setting up connection resilience...")
    
    # Set environment variables for the bridge
    os.environ['MEGAMIND_PROJECT_REALM'] = 'MegaMind_MCP'
    os.environ['MEGAMIND_PROJECT_NAME'] = 'MegaMind Context Database'
    os.environ['MEGAMIND_DEFAULT_TARGET'] = 'PROJECT'
    
    # Path to the actual bridge
    bridge_script = Path(__file__).parent / "stdio_http_bridge.py"
    
    try:
        # Start the bridge subprocess
        logger.info("Starting STDIO-HTTP bridge subprocess...")
        process = subprocess.Popen(
            [sys.executable, str(bridge_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,  # Unbuffered
            universal_newlines=True
        )
        
        logger.info("Bridge subprocess started, forwarding I/O...")
        
        # Forward stdin to subprocess
        def forward_stdin():
            try:
                for line in sys.stdin:
                    logger.debug(f"Forwarding to bridge: {line.strip()[:100]}...")
                    process.stdin.write(line)
                    process.stdin.flush()
            except (BrokenPipeError, EOFError):
                logger.info("Stdin closed, stopping forward")
            finally:
                if process.stdin and not process.stdin.closed:
                    process.stdin.close()
        
        # Forward subprocess stdout to stdout
        def forward_stdout():
            try:
                for line in iter(process.stdout.readline, ''):
                    if not line:
                        break
                    logger.debug(f"Response from bridge: {line.strip()[:100]}...")
                    sys.stdout.write(line)
                    sys.stdout.flush()
            except (BrokenPipeError, EOFError):
                logger.info("Stdout closed, stopping forward")
        
        # Start forwarding in threads
        import threading
        stdin_thread = threading.Thread(target=forward_stdin, daemon=True)
        stdout_thread = threading.Thread(target=forward_stdout, daemon=True)
        
        stdin_thread.start()
        stdout_thread.start()
        
        # Wait for process to complete
        return_code = process.wait()
        logger.info(f"Bridge subprocess completed with code: {return_code}")
        
        # Capture any stderr output
        stderr_output = process.stderr.read()
        if stderr_output:
            logger.info(f"Bridge stderr: {stderr_output}")
        
        return return_code
        
    except Exception as e:
        logger.error(f"Error in connector: {e}")
        return 1
    
    finally:
        if 'process' in locals() and process.poll() is None:
            logger.info("Terminating bridge subprocess...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Force killing bridge subprocess...")
                process.kill()

if __name__ == "__main__":
    sys.exit(main())