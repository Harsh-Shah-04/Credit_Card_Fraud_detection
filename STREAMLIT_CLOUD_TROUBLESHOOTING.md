# 🔧 Streamlit Cloud Troubleshooting Guide

If you encounter a `ModuleNotFoundError` when deploying to Streamlit Cloud, follow these steps:

## Issue: ModuleNotFoundError: No module named 'joblib'

### Causes
1. **Missing or incorrect requirements.txt** - Streamlit Cloud only installs packages listed in requirements.txt
2. **Version conflicts** - Some package versions may not be compatible
3. **Package name typos** - Ensure all packages are spelled correctly

### Solutions

#### Solution 1: Update requirements.txt (Recommended)
The updated `requirements.txt` uses version ranges for better compatibility:

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.21.0
scikit-learn>=1.0.0
tensorflow>=2.13.0
joblib>=1.3.0
matplotlib>=3.5.0
seaborn>=0.12.0
protobuf<=3.20.0
```

**Steps:**
1. Ensure your `requirements.txt` matches the above
2. Commit changes: `git add requirements.txt && git commit -m "Update requirements"`
3. Push to GitHub: `git push`
4. Streamlit Cloud will auto-deploy with new requirements

#### Solution 2: Use Streamlit Cloud Optimized Requirements
If the standard requirements don't work:

1. Delete `requirements.txt` content
2. Copy content from `streamlit_cloud_requirements.txt`
3. Rename to `requirements.txt`
4. Push to GitHub

#### Solution 3: Manual Redeployment
1. Go to your app on https://share.streamlit.io
2. Click "Manage app" (top right)
3. Click "Reboot app"
4. Streamlit will re-install all dependencies

#### Solution 4: Clear Cache
1. Go to your app on Streamlit Cloud
2. Click "Manage app" → "Settings"
3. Click "Clear cache"
4. Reboot the app

## Verification Steps

### Before Deploying
Test locally:
```bash
python test_imports.py
```

If all imports pass ✅, your app should work on Streamlit Cloud.

### After Deploying
1. Check the app logs:
   - Go to "Manage app" → "Logs"
   - Look for error messages
   
2. Common errors and fixes:

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: joblib` | joblib not in requirements.txt | Add `joblib>=1.3.0` to requirements.txt |
| `ModuleNotFoundError: tensorflow` | TensorFlow not installed | Add `tensorflow>=2.13.0` to requirements.txt |
| Version conflicts | Incompatible package versions | Use version ranges instead of pinned versions |
| Out of memory | Large model files | Ensure models are <500MB total |

## File Checklist for Cloud Deployment

Ensure these files exist in your repository:

- ✅ `streamlit_app.py` - Main application file
- ✅ `requirements.txt` - All dependencies listed
- ✅ `models/` - Directory with model files
  - ✅ `ann_model.h5`
  - ✅ `best_ml_model.pkl`
  - ✅ `scaler.pkl`
  - ✅ `ann_threshold.pkl`
- ✅ `sample_batch.csv` - Sample data
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `.gitignore` - Ignore unnecessary files

## Streamlit Cloud Configuration

### Create streamlit/config.toml

```toml
[theme]
primaryColor = "#1f77d3"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[client]
showErrorDetails = true
toolbarMode = "viewer"

[server]
maxUploadSize = 200
headless = true
enableCORS = false

[logger]
level = "warning"
```

## Complete Deployment Checklist

- [ ] Clone/download latest code
- [ ] Verify `requirements.txt` has all dependencies
- [ ] Run `python test_imports.py` locally ✅
- [ ] Commit all changes
- [ ] Push to GitHub: `git push origin main`
- [ ] Go to https://share.streamlit.io
- [ ] Create new app or reboot existing:
  - Repository: Your GitHub repo
  - Branch: `main`
  - Main file: `streamlit_app.py`
- [ ] Monitor deployment (check logs if errors)
- [ ] Test the app once it loads
- [ ] Share your public URL!

## Advanced Troubleshooting

### If you still get errors after trying above:

1. **Check model file sizes:**
   ```bash
   ls -lh models/
   ```
   Total should be < 1GB

2. **Verify model file integrity:**
   - Ensure .h5 and .pkl files are not corrupted
   - Can load them locally with joblib/TensorFlow

3. **Check Python version:**
   - Streamlit Cloud uses Python 3.11
   - Ensure your code is Python 3.11 compatible

4. **Review full logs:**
   - Manage app → Logs
   - Look for specific error messages
   - Search Google for the exact error

5. **Try minimal test:**
   - Create simple test app with just imports
   - Gradually add features
   - Identify which feature breaks the app

## Quick Command Reference

```bash
# Test imports locally
python test_imports.py

# Install all dependencies
pip install -r requirements.txt

# Run app locally
streamlit run streamlit_app.py

# Commit and push (after fixing)
git add .
git commit -m "Fix: Update dependencies for Streamlit Cloud"
git push origin main
```

## Success Indicators

✅ App loads without errors  
✅ All UI elements display  
✅ Single prediction works  
✅ Batch CSV upload works  
✅ Model info displays correctly  

## Get Help

If you're still stuck:

1. **Check Streamlit docs:** https://docs.streamlit.io/
2. **Check deployment docs:** https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app
3. **GitHub issues:** Search for similar issues
4. **Streamlit Community:** https://discuss.streamlit.io/

---

**Remember:** Any changes to `requirements.txt` will trigger a redeployment on Streamlit Cloud. The dependency installation happens automatically!

Good luck! 🚀
