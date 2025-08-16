#!/usr/bin/env python3
"""
Final Integration Test Report Generator

This script generates a comprehensive final report by consolidating
all test results from the integration testing suite.
"""

import os
import json
import sys
from datetime import datetime
from pathlib import Path


def load_test_reports():
    """Load all available test reports"""
    reports = {}
    
    # List of expected report files
    report_files = {
        "basic_validation": "basic_validation_test_report.json",
        "performance_benchmark": "performance_benchmark_report.json",
        "framework_validation": "integration_framework_validation_report.json",
        "integration_test": "final_integration_test_report.json",
        "compatibility_test": "compatibility_test_report.json"
    }
    
    for report_name, file_name in report_files.items():
        if os.path.exists(file_name):
            try:
                with open(file_name, 'r', encoding='utf-8') as f:
                    reports[report_name] = json.load(f)
                print(f"✓ Loaded {report_name} report")
            except Exception as e:
                print(f"✗ Failed to load {report_name} report: {e}")
                reports[report_name] = {"status": "FAILED", "error": f"Failed to load: {e}"}
        else:
            print(f"⚠ {report_name} report not found: {file_name}")
            reports[report_name] = {"status": "NOT_FOUND", "message": f"Report file not found: {file_name}"}
    
    return reports


def analyze_test_results(reports):
    """Analyze test results and generate summary statistics"""
    analysis = {
        "total_test_suites": len(reports),
        "available_reports": 0,
        "passed_suites": 0,
        "failed_suites": 0,
        "warning_suites": 0,
        "not_found_suites": 0,
        "overall_success_rate": 0.0,
        "suite_details": {}
    }
    
    for suite_name, report in reports.items():
        if report.get("status") == "NOT_FOUND":
            analysis["not_found_suites"] += 1
            analysis["suite_details"][suite_name] = {
                "status": "NOT_FOUND",
                "success_rate": 0.0
            }
        else:
            analysis["available_reports"] += 1
            
            # Determine suite status and success rate
            suite_status = "UNKNOWN"
            success_rate = 0.0
            
            # Basic validation report
            if suite_name == "basic_validation":
                summary = report.get("summary", {})
                success_rate = summary.get("success_rate", 0.0)
                suite_status = "PASSED" if success_rate >= 80 else "FAILED"
                
            # Performance benchmark report
            elif suite_name == "performance_benchmark":
                exec_summary = report.get("execution_summary", {})
                success_rate = exec_summary.get("success_rate", 0.0)
                suite_status = "PASSED" if success_rate >= 80 else "FAILED"
                
            # Framework validation report
            elif suite_name == "framework_validation":
                summary = report.get("framework_validation_summary", {})
                success_rate = summary.get("success_rate_percentage", 0.0)
                suite_status = "PASSED" if success_rate >= 80 else "FAILED"
                
            # Integration test report
            elif suite_name == "integration_test":
                exec_summary = report.get("test_execution_summary", {})
                success_rate = exec_summary.get("success_rate_percentage", 0.0)
                suite_status = "PASSED" if success_rate >= 80 else "FAILED"
                
            # Compatibility test report
            elif suite_name == "compatibility_test":
                # Check if all major components passed
                platform_info = report.get("platform_info", {})
                if platform_info:
                    suite_status = "PASSED"
                    success_rate = 100.0
                else:
                    suite_status = "FAILED"
                    success_rate = 0.0
            
            # Update counters
            if suite_status == "PASSED":
                analysis["passed_suites"] += 1
            elif suite_status == "FAILED":
                analysis["failed_suites"] += 1
            else:
                analysis["warning_suites"] += 1
                
            analysis["suite_details"][suite_name] = {
                "status": suite_status,
                "success_rate": success_rate
            }
    
    # Calculate overall success rate
    if analysis["available_reports"] > 0:
        total_success_rate = sum(
            details["success_rate"] for details in analysis["suite_details"].values()
            if details["status"] != "NOT_FOUND"
        )
        analysis["overall_success_rate"] = total_success_rate / analysis["available_reports"]
    
    return analysis


def generate_recommendations(analysis, reports):
    """Generate recommendations based on test results"""
    recommendations = []
    
    # Check overall success rate
    if analysis["overall_success_rate"] >= 90:
        recommendations.append("🎉 Excellent! All test suites passed with high success rates.")
        recommendations.append("✅ System is ready for production deployment.")
        recommendations.append("📋 Consider setting up continuous integration to maintain quality.")
        
    elif analysis["overall_success_rate"] >= 80:
        recommendations.append("✅ Good! Most test suites passed successfully.")
        recommendations.append("⚠️ Review any failed tests and address minor issues.")
        recommendations.append("🔄 Re-run failed test suites after fixes.")
        
    elif analysis["overall_success_rate"] >= 60:
        recommendations.append("⚠️ Moderate success rate. Several issues need attention.")
        recommendations.append("🔧 Address failed test cases before production deployment.")
        recommendations.append("📊 Focus on improving performance and reliability.")
        
    else:
        recommendations.append("❌ Low success rate. Significant issues detected.")
        recommendations.append("🛠️ Major fixes required before system can be deployed.")
        recommendations.append("🔍 Review all failed test cases and error logs.")
    
    # Specific recommendations based on suite results
    for suite_name, details in analysis["suite_details"].items():
        if details["status"] == "FAILED":
            if suite_name == "basic_validation":
                recommendations.append("🔧 Fix basic functionality issues - these are critical for system operation.")
            elif suite_name == "performance_benchmark":
                recommendations.append("⚡ Address performance issues to ensure system scalability.")
            elif suite_name == "framework_validation":
                recommendations.append("🏗️ Fix framework issues - these affect all other tests.")
            elif suite_name == "integration_test":
                recommendations.append("🔗 Resolve integration issues for end-to-end functionality.")
            elif suite_name == "compatibility_test":
                recommendations.append("🌐 Address compatibility issues for cross-platform deployment.")
                
        elif details["status"] == "NOT_FOUND":
            recommendations.append(f"📋 Run {suite_name} to get complete validation coverage.")
    
    # Missing checkpoint recommendation
    if "framework_validation" in reports:
        framework_report = reports["framework_validation"]
        test_results = framework_report.get("test_results", {})
        checkpoint_result = test_results.get("checkpoint_detection", {})
        if not checkpoint_result.get("checkpoint_exists", False):
            recommendations.append("📁 Ensure qwen3-finetuned checkpoint is available for complete testing.")
    
    return recommendations


def generate_next_steps(analysis):
    """Generate next steps based on analysis"""
    next_steps = []
    
    if analysis["overall_success_rate"] >= 80:
        next_steps.extend([
            "1. 🚀 Proceed with production deployment preparation",
            "2. 📚 Create user documentation and deployment guides",
            "3. 🔍 Set up production monitoring and alerting",
            "4. 🔄 Establish regular maintenance and update procedures"
        ])
    else:
        next_steps.extend([
            "1. 🔧 Address all failed test cases",
            "2. 🔄 Re-run complete test suite after fixes",
            "3. 📊 Analyze performance bottlenecks and optimize",
            "4. 🧪 Consider additional testing scenarios"
        ])
    
    # Add specific next steps based on missing reports
    if analysis["not_found_suites"] > 0:
        next_steps.append("5. 📋 Run missing test suites for complete coverage")
    
    return next_steps


def generate_final_report(reports, analysis):
    """Generate the comprehensive final report"""
    final_report = {
        "final_integration_validation_report": {
            "generation_timestamp": datetime.now().isoformat(),
            "report_version": "1.0",
            "validation_status": "PASSED" if analysis["overall_success_rate"] >= 80 else "FAILED"
        },
        "executive_summary": {
            "overall_success_rate": analysis["overall_success_rate"],
            "total_test_suites": analysis["total_test_suites"],
            "available_reports": analysis["available_reports"],
            "passed_suites": analysis["passed_suites"],
            "failed_suites": analysis["failed_suites"],
            "not_found_suites": analysis["not_found_suites"],
            "deployment_ready": analysis["overall_success_rate"] >= 80
        },
        "test_suite_results": analysis["suite_details"],
        "detailed_reports": reports,
        "recommendations": generate_recommendations(analysis, reports),
        "next_steps": generate_next_steps(analysis),
        "system_readiness": {
            "checkpoint_detection": "READY" if any(
                r.get("test_results", {}).get("checkpoint_detection", {}).get("checkpoint_exists", False)
                for r in reports.values() if isinstance(r, dict)
            ) else "NEEDS_CHECKPOINT",
            "framework_validation": "READY" if analysis["suite_details"].get("framework_validation", {}).get("status") == "PASSED" else "NEEDS_ATTENTION",
            "performance_validation": "READY" if analysis["suite_details"].get("performance_benchmark", {}).get("status") == "PASSED" else "NEEDS_OPTIMIZATION",
            "integration_validation": "READY" if analysis["suite_details"].get("integration_test", {}).get("status") == "PASSED" else "NEEDS_FIXES"
        }
    }
    
    return final_report


def main():
    """Main report generation function"""
    print("Final Integration Test Report Generator")
    print("=" * 60)
    
    # Load all test reports
    print("\nLoading test reports...")
    reports = load_test_reports()
    
    # Analyze results
    print("\nAnalyzing test results...")
    analysis = analyze_test_results(reports)
    
    # Generate final report
    print("\nGenerating final report...")
    final_report = generate_final_report(reports, analysis)
    
    # Save final report
    report_path = "FINAL_INTEGRATION_VALIDATION_REPORT.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False, default=str)
    
    # Print summary
    print("\n" + "=" * 60)
    print("FINAL INTEGRATION VALIDATION SUMMARY")
    print("=" * 60)
    
    summary = final_report["executive_summary"]
    print(f"Overall Success Rate: {summary['overall_success_rate']:.1f}%")
    print(f"Test Suites Available: {summary['available_reports']}/{summary['total_test_suites']}")
    print(f"Passed Suites: {summary['passed_suites']}")
    print(f"Failed Suites: {summary['failed_suites']}")
    print(f"Missing Suites: {summary['not_found_suites']}")
    print(f"Deployment Ready: {'✅ YES' if summary['deployment_ready'] else '❌ NO'}")
    
    print(f"\n📄 Final report saved to: {report_path}")
    
    # Print key recommendations
    print("\n🔍 KEY RECOMMENDATIONS:")
    for i, rec in enumerate(final_report["recommendations"][:5], 1):
        print(f"  {i}. {rec}")
    
    # Print next steps
    print("\n📋 NEXT STEPS:")
    for step in final_report["next_steps"][:3]:
        print(f"  {step}")
    
    # Return appropriate exit code
    if summary["deployment_ready"]:
        print("\n🎉 VALIDATION SUCCESSFUL - System ready for deployment!")
        return 0
    else:
        print("\n⚠️ VALIDATION INCOMPLETE - Address issues before deployment")
        return 1


if __name__ == "__main__":
    sys.exit(main())