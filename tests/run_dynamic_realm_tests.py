#!/usr/bin/env python3
"""
Automated Test Runner and Validation Framework
Phase 4.5 Implementation - Testing Strategy for Option 2 Execution Plan
"""

import argparse
import json
import os
import subprocess
import sys
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import unittest
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestResult:
    """Represents the result of a test run"""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.success: bool = False
        self.total_tests: int = 0
        self.passed_tests: int = 0
        self.failed_tests: int = 0
        self.error_tests: int = 0
        self.skipped_tests: int = 0
        self.error_messages: List[str] = []
        self.duration_seconds: float = 0.0
        
    def start(self):
        """Mark test start time"""
        self.start_time = datetime.now()
        
    def finish(self, success: bool, total: int = 0, passed: int = 0, 
              failed: int = 0, errors: int = 0, skipped: int = 0, 
              error_messages: List[str] = None):
        """Mark test completion with results"""
        self.end_time = datetime.now()
        self.success = success
        self.total_tests = total
        self.passed_tests = passed
        self.failed_tests = failed
        self.error_tests = errors
        self.skipped_tests = skipped
        self.error_messages = error_messages or []
        
        if self.start_time and self.end_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'test_name': self.test_name,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'success': self.success,
            'duration_seconds': self.duration_seconds,
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'error_tests': self.error_tests,
            'skipped_tests': self.skipped_tests,
            'error_messages': self.error_messages
        }

class DynamicRealmTestRunner:
    """Automated test runner for dynamic realm testing"""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.results: List[TestResult] = []
        self.overall_success = True
        
        # Test suite definitions
        self.test_suites = {
            'unit': {
                'name': 'Unit Tests',
                'description': 'Unit tests for dynamic realm components',
                'file': 'test_dynamic_realm_unit_tests.py',
                'timeout': 300,
                'required': True
            },
            'integration': {
                'name': 'Integration Tests',
                'description': 'End-to-end integration tests',
                'file': 'test_dynamic_realm_integration.py',
                'timeout': 600,
                'required': True
            },
            'security': {
                'name': 'Security Tests',
                'description': 'Security and malicious input tests',
                'file': 'test_dynamic_realm_security.py',
                'timeout': 900,
                'required': True
            },
            'performance': {
                'name': 'Performance Tests',
                'description': 'Performance and scalability tests',
                'file': 'test_dynamic_realm_performance.py',
                'timeout': 1200,
                'required': False
            },
            'phase3': {
                'name': 'Phase 3 Security Tests',
                'description': 'Phase 3 security pipeline tests',
                'file': 'test_phase3_security.py',
                'timeout': 600,
                'required': True
            }
        }
        
        # Environment checks
        self.environment_checks = [
            self._check_http_server,
            self._check_database_connection,
            self._check_python_dependencies,
            self._check_system_resources
        ]
    
    def run_all_tests(self, include_optional: bool = False, 
                     specific_suites: List[str] = None) -> bool:
        """Run all test suites"""
        logger.info("🚀 Starting Dynamic Realm Test Suite")
        logger.info(f"Base directory: {self.base_dir}")
        
        # Pre-flight environment checks
        if not self._run_environment_checks():
            logger.error("❌ Environment checks failed - aborting test run")
            return False
        
        # Determine which tests to run
        suites_to_run = self._determine_test_suites(include_optional, specific_suites)
        
        if not suites_to_run:
            logger.error("❌ No test suites to run")
            return False
        
        logger.info(f"Running {len(suites_to_run)} test suites: {', '.join(suites_to_run)}")
        
        # Run each test suite
        overall_start_time = datetime.now()
        
        for suite_name in suites_to_run:
            suite_config = self.test_suites[suite_name]
            result = self._run_test_suite(suite_name, suite_config)
            self.results.append(result)
            
            if not result.success and suite_config['required']:
                logger.error(f"❌ Required test suite '{suite_name}' failed - stopping test run")
                self.overall_success = False
                break
            elif not result.success:
                logger.warning(f"⚠️ Optional test suite '{suite_name}' failed - continuing")
                
        overall_duration = (datetime.now() - overall_start_time).total_seconds()
        
        # Generate test report
        self._generate_test_report(overall_duration)
        
        return self.overall_success
    
    def _determine_test_suites(self, include_optional: bool, 
                              specific_suites: List[str] = None) -> List[str]:
        """Determine which test suites to run"""
        if specific_suites:
            # Validate specific suites
            invalid_suites = [s for s in specific_suites if s not in self.test_suites]
            if invalid_suites:
                logger.error(f"Invalid test suites: {invalid_suites}")
                logger.info(f"Available suites: {list(self.test_suites.keys())}")
                return []
            return specific_suites
        
        # Run required suites plus optional if requested
        suites = [name for name, config in self.test_suites.items() if config['required']]
        
        if include_optional:
            optional = [name for name, config in self.test_suites.items() if not config['required']]
            suites.extend(optional)
        
        return suites
    
    def _run_environment_checks(self) -> bool:
        """Run pre-flight environment checks"""
        logger.info("🔍 Running environment checks...")
        
        all_passed = True
        for check_func in self.environment_checks:
            try:
                check_name = check_func.__name__.replace('_check_', '').replace('_', ' ').title()
                logger.info(f"  Checking {check_name}...")
                
                success, message = check_func()
                if success:
                    logger.info(f"    ✅ {message}")
                else:
                    logger.error(f"    ❌ {message}")
                    all_passed = False
                    
            except Exception as e:
                logger.error(f"    ❌ Check failed with exception: {e}")
                all_passed = False
        
        if all_passed:
            logger.info("✅ All environment checks passed")
        else:
            logger.error("❌ Some environment checks failed")
            
        return all_passed
    
    def _check_http_server(self) -> Tuple[bool, str]:
        """Check if HTTP server is available"""
        try:
            import requests
            response = requests.get("http://10.255.250.22:8080/mcp/health", timeout=5)
            if response.status_code == 200:
                return True, "HTTP server is responding"
            else:
                return False, f"HTTP server returned status {response.status_code}"
        except Exception as e:
            return False, f"HTTP server not reachable: {e}"
    
    def _check_database_connection(self) -> Tuple[bool, str]:
        """Check database connectivity"""
        try:
            # Try to import and test database connection
            # This is a simplified check - in practice, you'd test actual DB connection
            import mysql.connector
            return True, "Database connection libraries available"
        except ImportError:
            return False, "Database connection libraries not available"
        except Exception as e:
            return False, f"Database check failed: {e}"
    
    def _check_python_dependencies(self) -> Tuple[bool, str]:
        """Check required Python dependencies"""
        required_packages = ['requests', 'unittest', 'psutil', 'json', 'threading']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            return False, f"Missing required packages: {missing_packages}"
        else:
            return True, "All required Python packages available"
    
    def _check_system_resources(self) -> Tuple[bool, str]:
        """Check system resource availability"""
        try:
            import psutil
            
            # Check memory
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                return False, f"High memory usage: {memory.percent:.1f}%"
            
            # Check CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 80:
                return False, f"High CPU usage: {cpu_percent:.1f}%"
            
            # Check disk space
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                return False, f"Low disk space: {disk.percent:.1f}% used"
            
            return True, f"System resources OK (CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%, Disk: {disk.percent:.1f}%)"
            
        except Exception as e:
            return False, f"Resource check failed: {e}"
    
    def _run_test_suite(self, suite_name: str, suite_config: Dict[str, Any]) -> TestResult:
        """Run a single test suite"""
        logger.info(f"🧪 Running {suite_config['name']}...")
        logger.info(f"   Description: {suite_config['description']}")
        
        result = TestResult(suite_name)
        result.start()
        
        test_file = self.base_dir / suite_config['file']
        
        if not test_file.exists():
            result.finish(False, error_messages=[f"Test file not found: {test_file}"])
            logger.error(f"❌ Test file not found: {test_file}")
            return result
        
        try:
            # Run the test using subprocess for isolation
            cmd = [sys.executable, str(test_file)]
            
            logger.info(f"   Executing: {' '.join(cmd)}")
            logger.info(f"   Timeout: {suite_config['timeout']} seconds")
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=suite_config['timeout'],
                cwd=str(self.base_dir)
            )
            
            # Parse test results from output
            success = process.returncode == 0
            stdout = process.stdout
            stderr = process.stderr
            
            # Try to extract test statistics from output
            total, passed, failed, errors, skipped = self._parse_test_output(stdout)
            
            error_messages = []
            if stderr:
                error_messages.append(f"STDERR: {stderr}")
            if not success and not stderr:
                error_messages.append(f"Test failed with return code {process.returncode}")
            
            result.finish(success, total, passed, failed, errors, skipped, error_messages)
            
            if success:
                logger.info(f"   ✅ {suite_config['name']} passed ({total} tests, {result.duration_seconds:.1f}s)")
            else:
                logger.error(f"   ❌ {suite_config['name']} failed ({failed + errors} failures/errors)")
                if error_messages:
                    for msg in error_messages[:3]:  # Show first 3 error messages
                        logger.error(f"      {msg}")
            
        except subprocess.TimeoutExpired:
            result.finish(False, error_messages=[f"Test timed out after {suite_config['timeout']} seconds"])
            logger.error(f"❌ {suite_config['name']} timed out")
            
        except Exception as e:
            result.finish(False, error_messages=[f"Test execution failed: {e}"])
            logger.error(f"❌ {suite_config['name']} execution failed: {e}")
        
        return result
    
    def _parse_test_output(self, output: str) -> Tuple[int, int, int, int, int]:
        """Parse test output to extract statistics"""
        total = passed = failed = errors = skipped = 0
        
        lines = output.split('\n')
        
        # Look for unittest output patterns
        for line in lines:
            line = line.strip()
            
            # Pattern: "Ran X tests in Ys"
            if line.startswith('Ran ') and ' tests in ' in line:
                try:
                    total = int(line.split()[1])
                except (IndexError, ValueError):
                    pass
            
            # Pattern: "FAILED (failures=X, errors=Y)"
            elif line.startswith('FAILED '):
                try:
                    # Parse failures and errors
                    if 'failures=' in line:
                        failed_part = line.split('failures=')[1].split(',')[0].split(')')[0]
                        failed = int(failed_part)
                    if 'errors=' in line:
                        errors_part = line.split('errors=')[1].split(',')[0].split(')')[0]
                        errors = int(errors_part)
                except (IndexError, ValueError):
                    pass
            
            # Pattern: "OK" or success indicators
            elif line in ['OK', 'All tests passed']:
                passed = total
        
        # Calculate passed if not explicitly found
        if total > 0 and passed == 0 and failed + errors < total:
            passed = total - failed - errors - skipped
        
        return total, passed, failed, errors, skipped
    
    def _generate_test_report(self, overall_duration: float):
        """Generate comprehensive test report"""
        logger.info("\n" + "=" * 70)
        logger.info("🧪 DYNAMIC REALM TEST SUITE REPORT")
        logger.info("=" * 70)
        
        # Overall statistics
        total_suites = len(self.results)
        successful_suites = sum(1 for r in self.results if r.success)
        failed_suites = total_suites - successful_suites
        
        total_tests = sum(r.total_tests for r in self.results)
        total_passed = sum(r.passed_tests for r in self.results)
        total_failed = sum(r.failed_tests for r in self.results)
        total_errors = sum(r.error_tests for r in self.results)
        total_skipped = sum(r.skipped_tests for r in self.results)
        
        logger.info(f"📊 OVERALL SUMMARY:")
        logger.info(f"   Test Suites: {successful_suites}/{total_suites} passed")
        logger.info(f"   Individual Tests: {total_passed}/{total_tests} passed")
        logger.info(f"   Failed: {total_failed}, Errors: {total_errors}, Skipped: {total_skipped}")
        logger.info(f"   Duration: {overall_duration:.1f} seconds")
        logger.info(f"   Success Rate: {(total_passed / total_tests * 100):.1f}%" if total_tests > 0 else "   Success Rate: N/A")
        
        # Individual suite results
        logger.info(f"\n📋 SUITE DETAILS:")
        for result in self.results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            suite_config = self.test_suites[result.test_name]
            
            logger.info(f"   {status} {suite_config['name']}")
            logger.info(f"      Duration: {result.duration_seconds:.1f}s")
            logger.info(f"      Tests: {result.passed_tests}/{result.total_tests} passed")
            
            if not result.success and result.error_messages:
                logger.info(f"      Errors: {len(result.error_messages)} error(s)")
        
        # Generate JSON report
        self._generate_json_report(overall_duration)
        
        # Final status
        logger.info(f"\n🎯 FINAL RESULT:")
        if self.overall_success:
            logger.info("✅ ALL TESTS PASSED - Dynamic Realm System Ready")
        else:
            logger.error("❌ SOME TESTS FAILED - Review Issues Before Deployment")
            
        logger.info("=" * 70)
    
    def _generate_json_report(self, overall_duration: float):
        """Generate JSON test report for CI/CD integration"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_duration_seconds': overall_duration,
            'overall_success': self.overall_success,
            'summary': {
                'total_suites': len(self.results),
                'successful_suites': sum(1 for r in self.results if r.success),
                'total_tests': sum(r.total_tests for r in self.results),
                'passed_tests': sum(r.passed_tests for r in self.results),
                'failed_tests': sum(r.failed_tests for r in self.results),
                'error_tests': sum(r.error_tests for r in self.results),
                'skipped_tests': sum(r.skipped_tests for r in self.results)
            },
            'suites': [result.to_dict() for result in self.results]
        }
        
        report_file = self.base_dir / 'test_report.json'
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"📄 JSON report saved: {report_file}")
        except Exception as e:
            logger.warning(f"Failed to save JSON report: {e}")

def main():
    """Main entry point for test runner"""
    parser = argparse.ArgumentParser(
        description='Automated Test Runner for Dynamic Realm System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all required tests
  python run_dynamic_realm_tests.py
  
  # Run all tests including optional performance tests
  python run_dynamic_realm_tests.py --include-optional
  
  # Run specific test suites
  python run_dynamic_realm_tests.py --suites unit integration security
  
  # Run with verbose output
  python run_dynamic_realm_tests.py --verbose
        """
    )
    
    parser.add_argument(
        '--include-optional',
        action='store_true',
        help='Include optional test suites (e.g., performance tests)'
    )
    
    parser.add_argument(
        '--suites',
        nargs='+',
        choices=['unit', 'integration', 'security', 'performance', 'phase3'],
        help='Specific test suites to run'
    )
    
    parser.add_argument(
        '--base-dir',
        type=str,
        help='Base directory for test files (default: script directory)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--list-suites',
        action='store_true',
        help='List available test suites and exit'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create test runner
    runner = DynamicRealmTestRunner(args.base_dir)
    
    # List suites if requested
    if args.list_suites:
        print("Available test suites:")
        for name, config in runner.test_suites.items():
            required_text = "required" if config['required'] else "optional"
            print(f"  {name:12} - {config['description']} ({required_text})")
        return 0
    
    # Run tests
    try:
        success = runner.run_all_tests(
            include_optional=args.include_optional,
            specific_suites=args.suites
        )
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Test run interrupted by user")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"❌ Test runner failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())