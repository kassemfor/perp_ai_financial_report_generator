#!/usr/bin/env python3
"""Verify all dependencies are installed correctly."""

import sys
from importlib import import_module

REQUIRED_PACKAGES = [
    ('openai', 'OpenAI API'),
    ('google.generativeai', 'Google Gemini'),
    ('ollama', 'Ollama'),
    ('anthropic', 'Anthropic Claude'),
    ('pandas', 'Pandas'),
    ('openpyxl', 'OpenPyXL'),
    ('docx', 'python-docx'),
    ('PyPDF2', 'PyPDF2'),
    ('pdfplumber', 'pdfplumber'),
    ('streamlit', 'Streamlit'),
    ('reportlab', 'ReportLab'),
    ('numpy', 'NumPy'),
    ('pydantic', 'Pydantic'),
    ('dotenv', 'python-dotenv'),
    ('requests', 'Requests'),
    ('pytest', 'pytest'),
    ('yaml', 'PyYAML'),
    ('PIL', 'Pillow'),
    ('matplotlib', 'Matplotlib'),
]

def check_python_version():
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ ERROR: Python 3.8+ required")
        return False
    print("✓ Python version OK")
    return True

def check_packages():
    failed = []
    for package, name in REQUIRED_PACKAGES:
        try:
            import_module(package)
            print(f"✓ {name}")
        except ImportError as e:
            print(f"❌ {name} - NOT INSTALLED")
            failed.append((name, str(e)))
    return failed

def main():
    print("=" * 60)
    print("DEPENDENCY CHECK - AI Financial Report Generator")
    print("=" * 60)
    print()

    if not check_python_version():
        sys.exit(1)

    print()
    print("Checking installed packages...")
    print("-" * 60)
    failed = check_packages()

    print()
    print("=" * 60)
    if failed:
        print(f"❌ FAILED - {len(failed)} package(s) missing or invalid")
        print()
        print("Missing packages:")
        for name, err in failed:
            print(f"  • {name}")
        print()
        print("FIX: Run this command:")
        print("  pip install --no-cache-dir -r requirements_FIXED.txt")
        print()
        sys.exit(1)
    else:
        print("✅ SUCCESS - ALL DEPENDENCIES INSTALLED")
        print()
        print("Ready to run:")
        print("  streamlit run src/frontend/streamlit_app.py")
        print()
        sys.exit(0)

if __name__ == "__main__":
    main()
