"""
start_services.py - Unified Industrial Services Orchestrator
SIH26170: Space-Grade Semiconductor Latent Defect Screening

Launches:
1. FastAPI Enterprise REST & WebSocket Backend (Port 8000)
2. Streamlit Mission Control & Engineering Dashboard (Port 8501)
"""

import sys
import subprocess
import os
import time


def main():
    print("=" * 65)
    print("SIH26170: SPACE-GRADE BURN-IN LATENT DEFECT SCREENING")
    print("Automated Multi-Lot Parametric Screening & Real-Time Integrations")
    print("=" * 65)
    
    python_exe = sys.executable
    project_root = os.path.abspath(os.path.dirname(__file__))
    
    print("\nSelect execution mode:")
    print("1) Launch Full Suite (FastAPI Backend :8000 + Streamlit Dashboard :8501)")
    print("2) Launch FastAPI REST API Server Only (:8000)")
    print("3) Launch Streamlit Mission Dashboard Only (:8501)")
    print("4) Run All Automated Unit & Integration Tests")
    
    choice = sys.argv[1] if len(sys.argv) > 1 else "1"
    
    if choice == "2":
        print("\nStarting FastAPI REST Server on http://localhost:8000 ...")
        cmd = [python_exe, os.path.join(project_root, "run_api.py")]
        subprocess.run(cmd)
        
    elif choice == "3":
        print("\nStarting Streamlit Mission Dashboard on http://localhost:8501 ...")
        cmd = [python_exe, "-m", "streamlit", "run", os.path.join(project_root, "app", "streamlit_app.py")]
        subprocess.run(cmd)
        
    elif choice == "4":
        print("\nExecuting Comprehensive Automated Test Suite ...")
        cmd = [python_exe, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]
        subprocess.run(cmd)
        
    else:
        print("\nStarting FastAPI REST Server in background on :8000 ...")
        p_api = subprocess.Popen([python_exe, os.path.join(project_root, "run_api.py")])
        time.sleep(2)
        
        print("Starting Streamlit Dashboard on :8501 ...")
        try:
            cmd = [python_exe, "-m", "streamlit", "run", os.path.join(project_root, "app", "streamlit_app.py")]
            subprocess.run(cmd)
        finally:
            print("\nShutting down FastAPI background server...")
            p_api.terminate()
            p_api.wait()


if __name__ == "__main__":
    main()
