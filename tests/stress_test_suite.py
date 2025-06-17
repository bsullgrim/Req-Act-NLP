"""
Comprehensive Stress Test Suite for Requirements Analyzer
Tests edge cases, performance, encoding, and error handling
"""

import pandas as pd
import numpy as np
import time
import random
import string
import tempfile
import os
from pathlib import Path
import sys
import traceback
from typing import List, Dict, Tuple
import concurrent.futures
import psutil
import memory_profiler

# Add project root for imports
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.quality.reqGrading import RequirementAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    print("⚠️ Could not import RequirementAnalyzer - some tests will be skipped")
    ANALYZER_AVAILABLE = False

class StressTestSuite:
    """Comprehensive stress testing for Requirements Analyzer"""
    
    def __init__(self):
        self.results = []
        self.test_files = []
        self.cleanup_files = []
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def cleanup(self):
        """Clean up temporary test files"""
        for file_path in self.cleanup_files:
            try:
                if Path(file_path).exists():
                    os.unlink(file_path)
            except Exception as e:
                print(f"⚠️ Could not clean up {file_path}: {e}")
    
    def log_result(self, test_name: str, success: bool, duration: float, 
                   details: str = "", memory_mb: float = 0):
        """Log test result"""
        self.results.append({
            'test': test_name,
            'success': success,
            'duration_sec': duration,
            'memory_mb': memory_mb,
            'details': details
        })
        
        status = "✅" if success else "❌"
        print(f"{status} {test_name}: {duration:.2f}s {details}")
    
    def generate_edge_case_requirements(self) -> List[Tuple[str, str]]:
        """Generate problematic requirements for edge case testing"""
        
        edge_cases = [
            # Encoding edge cases
            ("encoding_unicode", "The system shall handle émojis 🚀 and ñoñó characters properly"),
            ("encoding_special", "The system shall process data with symbols: ±∞≤≥αβγ∑∏∆"),
            ("encoding_mixed", "System shall support ASCII, UTF-8: café, naïve, résumé"),
            
            # Length edge cases  
            ("empty", ""),
            ("whitespace_only", "   \t\n   "),
            ("single_char", "A"),
            ("ultra_short", "Go"),
            ("ultra_long", " ".join(["The system shall"] + ["do something"] * 200)),
            
            # Special characters
            ("quotes_mixed", 'The system shall handle "quotes" and \'apostrophes\' correctly'),
            ("symbols_heavy", "System shall process @#$%^&*(){}[]|\\:;\"'<>,.?/~`"),
            ("newlines", "The system shall\nhandle\nmultiple\nlines"),
            ("tabs", "The\tsystem\tshall\thandle\ttabs"),
            
            # Number edge cases
            ("sci_notation", "Maintain 1e-6 g, 2.5E+10 Hz, and 3.14159×10^-12 accuracy"),
            ("large_numbers", "Process 999999999999999999999 operations per second"),
            ("negative_numbers", "Operate between -273.15°C and -40°F"),
            ("fraction_heavy", "Achieve 1/3, 2/7, 22/7, and π accuracy"),
            
            # Malformed requirements
            ("no_modal", "System display status to users"),
            ("only_modal", "shall shall shall must will"),
            ("backwards", "users to status display shall system The"),
            ("repeated_words", "The the system system shall shall do do something something"),
            
            # Extreme ambiguity
            ("all_ambiguous", "The system shall appropriately handle good stuff efficiently as needed"),
            ("no_nouns", "Shall quickly and efficiently process and transmit and receive"),
            ("no_verbs", "The system good performance appropriate response"),
            
            # Technical edge cases
            ("units_mixed", "Handle 5kg, 10 pounds, 3.2m, 50 inches, 20°C, 68°F simultaneously"),
            ("standards_soup", "Comply with ISO-9001 IEEE-802.11 MIL-STD-810G DO-178B RTCA"),
            ("version_numbers", "Support v1.2.3, v2.0.0-alpha, v3.1.4-beta.2+build.123"),
            
            # Linguistic nightmares
            ("nested_clauses", "The system, which shall be designed to operate in environments that may include conditions where temperatures can range from those that might be considered cold to those that could be deemed warm, shall function"),
            ("passive_chain", "Data is processed by modules that are controlled by systems that are managed by operators"),
            ("conjunction_overload", "System shall input and process and validate and store and retrieve and display and print and export and backup and archive data"),
            
            # Domain-specific stress
            ("aerospace_jargon", "The system shall maintain LEO orbital mechanics with ΔV calculations for Hohmann transfers"),
            ("medical_terms", "Process electrocardiographic, electroencephalographic, and magnetoencephalographic signals"),
            ("legal_language", "Shall comply with all applicable federal, state, local, and international regulations heretofore and hereafter enacted"),
            
            # NLP breaking attempts
            ("homonyms", "The system shall read the read data and lead users to lead decisions"),
            ("context_switch", "Bank on the river bank to bank the data in the memory bank"),
            ("acronym_overload", "The API shall interface with REST, SOAP, XML, JSON, HTTP, HTTPS, TCP, UDP, IP protocols"),
        ]
        
        return edge_cases
    
    def generate_large_dataset(self, size: int = 10000) -> pd.DataFrame:
        """Generate large dataset for performance testing"""
        
        requirement_templates = [
            "The system shall {verb} {object} {condition}",
            "The {component} shall {action} with {criteria}",
            "During {phase}, the system shall {behavior}",
            "{Actor} shall be able to {capability}",
            "The system shall {performance} within {timeframe}",
        ]
        
        verbs = ["process", "handle", "manage", "control", "monitor", "analyze", "generate", "validate"]
        objects = ["data", "requests", "signals", "information", "commands", "responses", "files", "messages"]
        conditions = ["within 5 seconds", "with 99% accuracy", "under normal conditions", "as required"]
        components = ["interface", "module", "subsystem", "controller", "processor", "sensor", "actuator"]
        actions = ["respond", "operate", "function", "perform", "execute", "communicate", "calculate"]
        criteria = ["high precision", "low latency", "minimal error", "maximum efficiency", "optimal performance"]
        phases = ["startup", "operation", "shutdown", "maintenance", "emergency", "testing"]
        actors = ["Users", "Operators", "Administrators", "The system", "External systems"]
        capabilities = ["access data", "modify settings", "view reports", "generate alerts", "perform diagnostics"]
        performance = ["complete processing", "respond to requests", "update displays", "synchronize data"]
        timeframes = ["2 seconds", "100 milliseconds", "1 minute", "real-time", "5 minutes"]
        
        requirements = []
        for i in range(size):
            template = random.choice(requirement_templates)
            requirement = template.format(
                verb=random.choice(verbs),
                object=random.choice(objects),
                condition=random.choice(conditions),
                component=random.choice(components),
                action=random.choice(actions),
                criteria=random.choice(criteria),
                phase=random.choice(phases),
                Actor=random.choice(actors),
                capability=random.choice(capabilities),
                performance=random.choice(performance),
                timeframe=random.choice(timeframes)
            )
            requirements.append(requirement)
        
        return pd.DataFrame({
            'ID': [f'REQ_{i:05d}' for i in range(size)],
            'Name': [f'Requirement {i}' for i in range(size)],
            'Requirement Text': requirements
        })
    
    def test_edge_cases(self):
        """Test edge cases and malformed inputs"""
        if not ANALYZER_AVAILABLE:
            print("⚠️ Skipping edge case tests - analyzer not available")
            return
            
        print("\n🧪 EDGE CASE TESTING")
        print("=" * 50)
        
        edge_cases = self.generate_edge_case_requirements()
        analyzer = RequirementAnalyzer()
        
        for case_name, requirement_text in edge_cases:
            start_time = time.time()
            try:
                issues, metrics = analyzer.analyze_requirement(requirement_text, case_name)
                duration = time.time() - start_time
                
                # Check for reasonable results
                success = True
                details = f"Issues: {len(issues)}, Score: {metrics.clarity_score:.1f}"
                
                # Flag suspicious results
                if len(issues) == 0 and len(requirement_text.strip()) < 5:
                    success = False
                    details += " (SUSPICIOUS: No issues for very short text)"
                
                self.log_result(f"edge_case_{case_name}", success, duration, details)
                
            except Exception as e:
                duration = time.time() - start_time
                self.log_result(f"edge_case_{case_name}", False, duration, f"ERROR: {str(e)}")
    
    def test_performance_scaling(self):
        """Test performance with increasing dataset sizes"""
        if not ANALYZER_AVAILABLE:
            print("⚠️ Skipping performance tests - analyzer not available")
            return
            
        print("\n⚡ PERFORMANCE SCALING TESTS")
        print("=" * 50)
        
        sizes = [10, 50, 100, 500, 1000, 2000]
        analyzer = RequirementAnalyzer()
        
        for size in sizes:
            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024
            
            try:
                # Generate test data
                df = self.generate_large_dataset(size)
                
                # Create temporary CSV
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
                    df.to_csv(f.name, index=False)
                    temp_file = f.name
                    self.cleanup_files.append(temp_file)
                
                # Run analysis
                result_df = analyzer.analyze_file(temp_file, excel_report=False)
                
                duration = time.time() - start_time
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                memory_used = memory_after - memory_before
                
                # Calculate throughput
                throughput = size / duration if duration > 0 else 0
                
                success = len(result_df) == size
                details = f"{throughput:.1f} req/sec, {memory_used:.1f}MB"
                
                self.log_result(f"performance_{size}_reqs", success, duration, details, memory_used)
                
            except Exception as e:
                duration = time.time() - start_time
                self.log_result(f"performance_{size}_reqs", False, duration, f"ERROR: {str(e)}")
    
    def test_encoding_stress(self):
        """Test various encoding scenarios"""
        print("\n🌐 ENCODING STRESS TESTS")
        print("=" * 50)
        
        encoding_tests = [
            ('utf8_bom', 'utf-8-sig'),
            ('latin1', 'latin-1'), 
            ('cp1252', 'cp1252'),
            ('ascii', 'ascii'),
        ]
        
        test_data = pd.DataFrame({
            'ID': ['ENC_001', 'ENC_002', 'ENC_003'],
            'Requirement Text': [
                'System shall handle café and naïve text',
                'Process résumé data with proper encoding',
                'Support émojis 🚀 and symbols ±∞≤≥'
            ]
        })
        
        for enc_name, encoding in encoding_tests:
            start_time = time.time()
            try:
                # Create file with specific encoding
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding=encoding) as f:
                    test_data.to_csv(f.name, index=False, encoding=encoding)
                    temp_file = f.name
                    self.cleanup_files.append(temp_file)
                
                if ANALYZER_AVAILABLE:
                    analyzer = RequirementAnalyzer()
                    result_df = analyzer.analyze_file(temp_file, excel_report=False)
                    success = len(result_df) == len(test_data)
                else:
                    # Just test file creation/reading
                    test_read = pd.read_csv(temp_file, encoding=encoding)
                    success = len(test_read) == len(test_data)
                
                duration = time.time() - start_time
                self.log_result(f"encoding_{enc_name}", success, duration)
                
            except Exception as e:
                duration = time.time() - start_time
                self.log_result(f"encoding_{enc_name}", False, duration, f"ERROR: {str(e)}")
    
    def test_memory_usage(self):
        """Test memory usage patterns"""
        if not ANALYZER_AVAILABLE:
            print("⚠️ Skipping memory tests - analyzer not available")
            return
            
        print("\n🧠 MEMORY USAGE TESTS")
        print("=" * 50)
        
        @memory_profiler.profile
        def memory_intensive_analysis():
            analyzer = RequirementAnalyzer()
            large_df = self.generate_large_dataset(1000)
            
            # Analyze each requirement individually to stress memory
            for idx, row in large_df.iterrows():
                analyzer.analyze_requirement(row['Requirement Text'], row['ID'])
                
                # Force garbage collection every 100 requirements
                if idx % 100 == 0:
                    import gc
                    gc.collect()
        
        start_time = time.time()
        try:
            memory_intensive_analysis()
            duration = time.time() - start_time
            self.log_result("memory_stress_1000_reqs", True, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("memory_stress_1000_reqs", False, duration, f"ERROR: {str(e)}")
    
    def test_concurrent_access(self):
        """Test concurrent analyzer usage"""
        if not ANALYZER_AVAILABLE:
            print("⚠️ Skipping concurrency tests - analyzer not available")
            return
            
        print("\n🔄 CONCURRENCY TESTS")
        print("=" * 50)
        
        def worker_analysis(worker_id):
            try:
                analyzer = RequirementAnalyzer()
                test_df = self.generate_large_dataset(50)
                
                results = []
                for _, row in test_df.iterrows():
                    issues, metrics = analyzer.analyze_requirement(row['Requirement Text'])
                    results.append(len(issues))
                
                return worker_id, True, len(results)
            except Exception as e:
                return worker_id, False, str(e)
        
        start_time = time.time()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(worker_analysis, i) for i in range(4)]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
            
            duration = time.time() - start_time
            successful_workers = sum(1 for _, success, _ in results if success)
            
            success = successful_workers == 4
            details = f"{successful_workers}/4 workers succeeded"
            
            self.log_result("concurrent_4_workers", success, duration, details)
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("concurrent_4_workers", False, duration, f"ERROR: {str(e)}")
    
    def test_error_recovery(self):
        """Test error handling and recovery"""
        print("\n🚨 ERROR RECOVERY TESTS")
        print("=" * 50)
        
        error_scenarios = [
            ("missing_file", "nonexistent_file.csv"),
            ("empty_file", ""),
            ("malformed_csv", "not,a,proper\ncsv,file"),
            ("missing_column", "ID,Name\n1,Test"),  # Missing 'Requirement Text'
        ]
        
        for scenario_name, content in error_scenarios:
            start_time = time.time()
            try:
                # Create problematic file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    if scenario_name != "missing_file":
                        f.write(content)
                        temp_file = f.name
                        self.cleanup_files.append(temp_file)
                    else:
                        temp_file = content  # Use the nonexistent filename
                
                if ANALYZER_AVAILABLE:
                    analyzer = RequirementAnalyzer()
                    try:
                        result_df = analyzer.analyze_file(temp_file, excel_report=False)
                        # Should not reach here for error scenarios
                        success = False
                        details = "ERROR: Should have failed but didn't"
                    except Exception as expected_error:
                        # Expected behavior - graceful error handling
                        success = True
                        details = f"Gracefully handled: {type(expected_error).__name__}"
                else:
                    # Just test file operations
                    try:
                        pd.read_csv(temp_file)
                        success = False
                        details = "ERROR: Should have failed but didn't"
                    except Exception:
                        success = True
                        details = "Gracefully handled file error"
                
                duration = time.time() - start_time
                self.log_result(f"error_recovery_{scenario_name}", success, duration, details)
                
            except Exception as e:
                duration = time.time() - start_time
                self.log_result(f"error_recovery_{scenario_name}", False, duration, f"Unexpected error: {str(e)}")
    
    def run_all_tests(self):
        """Run complete stress test suite"""
        print("🚀 REQUIREMENTS ANALYZER STRESS TEST SUITE")
        print("=" * 70)
        print(f"Analyzer available: {ANALYZER_AVAILABLE}")
        print(f"System: {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total / 1024**3:.1f}GB RAM")
        print("=" * 70)
        
        start_time = time.time()
        
        # Run all test categories
        self.test_edge_cases()
        self.test_encoding_stress()
        self.test_error_recovery()
        
        if ANALYZER_AVAILABLE:
            self.test_performance_scaling()
            self.test_memory_usage()
            self.test_concurrent_access()
        
        total_duration = time.time() - start_time
        
        # Generate summary report
        self.generate_summary_report(total_duration)
    
    def generate_summary_report(self, total_duration: float):
        """Generate comprehensive test summary"""
        print("\n📊 STRESS TEST SUMMARY")
        print("=" * 50)
        
        df = pd.DataFrame(self.results)
        
        if len(df) == 0:
            print("No test results to report")
            return
        
        total_tests = len(df)
        passed_tests = len(df[df['success'] == True])
        failed_tests = total_tests - passed_tests
        
        avg_duration = df['duration_sec'].mean()
        max_duration = df['duration_sec'].max()
        total_memory = df['memory_mb'].sum()
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        print(f"Total Duration: {total_duration:.2f}s")
        print(f"Average Test Duration: {avg_duration:.3f}s")
        print(f"Longest Test: {max_duration:.2f}s")
        if total_memory > 0:
            print(f"Peak Memory Usage: {total_memory:.1f}MB")
        
        # Show failed tests
        if failed_tests > 0:
            print(f"\n❌ FAILED TESTS:")
            failed_df = df[df['success'] == False]
            for _, row in failed_df.iterrows():
                print(f"  {row['test']}: {row['details']}")
        
        # Show performance stats
        perf_tests = df[df['test'].str.contains('performance_')]
        if len(perf_tests) > 0:
            print(f"\n⚡ PERFORMANCE BREAKDOWN:")
            for _, row in perf_tests.iterrows():
                print(f"  {row['test']}: {row['duration_sec']:.2f}s ({row['details']})")
        
        print(f"\n🎯 Overall Result: {'PASS' if failed_tests == 0 else 'FAIL'}")

def main():
    """Run stress test suite"""
    with StressTestSuite() as test_suite:
        test_suite.run_all_tests()

if __name__ == "__main__":
    main()
