# 📦 INSTALLATION INSTRUCTIONS - FIXED VERSION

## One-Liner for Most Users

```bash
# Step 1: Create virtual environment
python3 -m venv venv && source venv/bin/activate

# Step 2: Install (all in one line)
pip install --upgrade pip && pip install --no-cache-dir -r requirements_FIXED.txt

# Step 3: Verify
python test_dependencies.py

# Step 4: Extract and run project
unzip ai_financial_report_generator_v1.0.zip
cd ai_financial_report_generator
streamlit run src/frontend/streamlit_app.py
```

## Step-by-Step Installation

### Windows
```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
pip install --no-cache-dir -r requirements_FIXED.txt

# Verify
python test_dependencies.py
```

### Linux / macOS
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
python3 -m pip install --upgrade pip
pip install --no-cache-dir -r requirements_FIXED.txt

# Verify
python test_dependencies.py
```

## What's Fixed

| Issue | Before | After |
|-------|--------|-------|
| openpyxl | ❌ 3.11.0 (doesn't exist) | ✓ 3.0.0-3.1.5 |
| numpy | ❌ Conflicted with pandas | ✓ <1.25.0 |
| pandas | ❌ v2.0+ incompatible | ✓ <2.1.0 |
| matplotlib | ❌ Breaking changes | ✓ <3.8.0 |

## Troubleshooting

**Q: "Command 'python3' not found"**
A: Install Python 3.8+ from https://www.python.org/downloads/

**Q: "pip: command not found"**
A: Run `python -m pip` instead of `pip`

**Q: Still getting version errors**
A: Try the Docker approach (no dependency issues)

**Q: Installation is very slow**
A: Add `--no-cache-dir` flag (already in commands above)

## Docker Option (No Setup Needed)

```bash
# Extract project
unzip ai_financial_report_generator_v1.0.zip
cd ai_financial_report_generator

# Run with Docker (if installed)
docker-compose up

# Access at http://localhost:8501
```

## Next Steps After Installation

1. ✓ Verify: `python test_dependencies.py`
2. Extract project: `unzip ai_financial_report_generator_v1.0.zip`
3. Enter directory: `cd ai_financial_report_generator`
4. Run: `streamlit run src/frontend/streamlit_app.py`
5. Open browser: http://localhost:8501

You're done! 🎉

## File Reference

- **requirements_FIXED.txt** - All verified, compatible versions
- **DEPENDENCY_FIXES_GUIDE.md** - Detailed troubleshooting
- **test_dependencies.py** - Verify installation
- **INSTALLATION_INSTRUCTIONS.md** - This file
