"""
Root entry point for Streamlit Community Cloud deployment.
"""
import runpy
import os
import sys

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Run the main dashboard script
runpy.run_path(os.path.join("dashboards", "dashboard.py"), run_name="__main__")
