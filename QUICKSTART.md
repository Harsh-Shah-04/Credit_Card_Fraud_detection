# 🚀 Quick Start Guide

## Get Your Credit Card Fraud Detection App Running in 2 Minutes!

### Option 1: One-Click Start (Easiest) 🎯

#### Windows Users:
1. Navigate to your project folder
2. **Double-click `run_app.bat`**
3. Wait for the app to open in your browser (http://localhost:8501)
   - Or run: `streamlit run streamlit_app.py`

✅ Done! You're ready to use the app!

#### MacOS/Linux Users:
```bash
cd "d:/Krishna/study/semester 6/ML_Innotex"
chmod +x run_app.sh
./run_app.sh
```

✅ Done! Check http://localhost:8501

---

### Option 2: Manual Setup

```bash
# 1. Navigate to project
cd "d:/Krishna/study/semester 6/ML_Innotex"

# 2. Create virtual environment (optional but recommended)
python -m venv venv

# 3. Activate it
# Windows:
venv\Scripts\activate
# MacOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run app
streamlit run streamlit_app.py
```

---

## 🎮 Using the App

### Home Page
- Overview of the system
- Model information
- Feature descriptions

### Single Prediction (🔍 Tab)
1. Enter transaction details:
   - Amount (e.g., $150.50)
   - Type (Online, In-Store, ATM, Transfer)
   - Time (Hour, Day, Month, Day of Week)
   - Location (city)
2. Click "Predict"
3. See if transaction is Fraudulent or Legitimate

### Batch Prediction (📊 Tab)
1. Download or use `sample_batch.csv` as example
2. Upload your CSV file
3. Click "Make Batch Predictions"
4. Download results

### Model Info (📈 Tab)
- Performance metrics
- Model comparison
- Architecture details

---

## 📊 Test with Sample Data

We've included a sample CSV file: `sample_batch.csv`

Try it out:
1. Go to "📊 Batch Prediction" tab
2. Click "Upload CSV file"
3. Select `sample_batch.csv`
4. Click "Make Batch Predictions"
5. Download your results!

---

## 🌐 Deploy Online (Choose One)

### FREE Option 1: Streamlit Cloud (Recommended) ⭐
1. Push code to GitHub:
   ```bash
   git add .
   git commit -m "Add fraud detection app"
   git push origin main
   ```
2. Go to https://share.streamlit.io
3. Click "New app"
4. Connect your GitHub repository
5. Done! Your app is live! 🎉

### FREE Option 2: Docker (if you know Docker)
```bash
docker-compose up --build
```
Access at http://localhost:8501

### PAID: Heroku
```bash
heroku create your-app-name
git push heroku main
```

For detailed deployment options, see `DEPLOYMENT.md`

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'streamlit'"
```bash
pip install -r requirements.txt
```

### Models not found
Make sure these files exist in `models/` folder:
- ✅ ann_model.h5
- ✅ best_ml_model.pkl
- ✅ scaler.pkl
- ✅ ann_threshold.pkl

### Port 8501 already in use
```bash
# Kill the process
# Windows: taskkill /PID <pid> /F
# MacOS/Linux: kill -9 <pid>

# Or use different port:
streamlit run app.py --server.port 8502
```

---

## 📚 File Structure

```
Your Project/
├── app.py                    ← Main app
├── requirements.txt          ← Dependencies
├── models/                   ← Trained models ✅ Already here!
│   ├── ann_model.h5
│   ├── best_ml_model.pkl
│   ├── scaler.pkl
│   └── ann_threshold.pkl
├── sample_batch.csv          ← Sample data for testing
├── README.md                 ← Full documentation
├── DEPLOYMENT.md             ← Deployment guide
├── Dockerfile                ← For Docker deployment
├── docker-compose.yml        ← Docker compose
├── run_app.bat               ← Windows launcher
└── run_app.sh                ← MacOS/Linux launcher
```

---

## 💡 Key Features

✅ **Real-time Prediction** - Get fraud probability instantly  
✅ **Batch Processing** - Process CSV files with multiple transactions  
✅ **High Accuracy** - 99.23% accuracy with ANN model  
✅ **User-Friendly Interface** - Beautiful, interactive UI  
✅ **Multiple Models** - ANN and Decision Tree available  
✅ **Production Ready** - Can be deployed to cloud  
✅ **Sample Data** - Included CSV for testing  

---

## 🎯 Next Steps

1. ✅ Run the app locally
2. ✅ Test with single transactions
3. ✅ Try batch prediction with sample_batch.csv
4. ✅ Review model performance metrics
5. ✅ Deploy to Streamlit Cloud or Docker
6. ✅ Share with stakeholders!

---

## ❓ Need Help?

- **For app issues**: Check `README.md`
- **For deployment**: Check `DEPLOYMENT.md`
- **For errors**: See Troubleshooting section above
- **For model details**: See model info in the app (📈 tab)

---

## 📞 Support Resources

- **Streamlit Docs**: https://docs.streamlit.io/
- **TensorFlow Docs**: https://www.tensorflow.org/
- **Scikit-learn Docs**: https://scikit-learn.org/
- **GitHub Issues**: Create an issue if you find bugs

---

## 🎉 You're All Set!

Your credit card fraud detection system is ready to use!

**Quick Commands Reference:**

| Task | Windows | MacOS/Linux |
|------|---------|------------|
| Start App | `run_app.bat` | `./run_app.sh` |
| Stop App | `Ctrl+C` | `Ctrl+C` |
| Change Port | Add `--server.port 8502` | `streamlit run app.py --server.port 8502` |
| Install Packages | `pip install -r requirements.txt` | `pip3 install -r requirements.txt` |

---

**Happy Predicting! 🚀**

If you find this helpful, star the repository ⭐

Last Updated: December 2024
