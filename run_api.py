"""
run_api.py - Entry point to launch the SIH26170 Enterprise REST API
"""
import uvicorn
import os
import sys

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting SIH26170 Space-Grade Screening API on http://{host}:{port}")
    uvicorn.run("src.integrations.api:app", host=host, port=port, reload=False, log_level="info")
