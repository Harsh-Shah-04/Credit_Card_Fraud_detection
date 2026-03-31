# 💳 Credit Card Fraud Detection - Streamlit Application

## Overview
This is a production-ready Streamlit application for real-time credit card fraud detection using a trained Artificial Neural Network (ANN) model. The application provides an intuitive interface for:
- Single transaction fraud prediction
- Batch processing of multiple transactions
- Model performance metrics and comparison
- Interactive visualizations and analysis

## Features

### 🔍 Single Prediction
- Input transaction details in real-time
- Instant fraud probability assessment
- Risk analysis with warning indicators
- Support for both ANN and ML models

### 📊 Batch Prediction
- Upload CSV files with multiple transactions
- Process multiple predictions simultaneously
- Export results with fraud probabilities
- Visualization of prediction distribution

### 📈 Model Information
- Detailed model performance comparison
- Architecture and training configuration details
- Feature descriptions and ranges
- Evaluation metrics (Accuracy, ROC-AUC, Precision, Recall, F1-Score)

## Model Architecture

### Artificial Neural Network (ANN) - Recommended
```
Input Layer: 15 features
    ↓
Dense(256, ReLU) → BatchNorm → Dropout(0.3)
    ↓
Dense(128, ReLU) → BatchNorm → Dropout(0.3)
    ↓
Dense(64, ReLU) → BatchNorm → Dropout(0.2)
    ↓
Dense(32, ReLU) → Dropout(0.2)
    ↓
Output Layer: Sigmoid (Binary Classification)
```

**Configuration:**
- Optimizer: Adam (learning_rate=0.001)
- Loss: Binary Crossentropy
- Regularization: Batch Normalization + Dropout
- Training Epochs: 60 with Early Stopping
- Callbacks: EarlyStopping (patience=10), ReduceLROnPlateau

### Model Performance
- **Accuracy**: 99.23%
- **ROC-AUC**: 0.9889
- **Precision**: 95.34%
- **Recall**: 89.12%
- **F1-Score**: 92.12%

## Input Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| Amount | Numerical | $0 - $10,000 | Transaction amount |
| TransactionType | Categorical | Online, In-Store, ATM, Transfer | Type of transaction |
| Hour | Numerical | 0-23 | Hour of transaction (24-hour format) |
| Day | Numerical | 1-31 | Day of month |
| Month | Numerical | 1-12 | Month of year |
| DayOfWeek | Numerical | 0-6 | Day of week (0=Monday, 6=Sunday) |
| Location | Categorical | 9 cities | Transaction location |

Supported Locations:
- Dallas
- Houston
- Los Angeles
- New York
- Philadelphia
- Phoenix
- San Antonio
- San Diego
- San Jose

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Clone/Download the Project
```bash
cd "d:\Krishna\study\semester 6\ML_Innotex"
```

### Step 2: Create Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# MacOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application Locally
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## Usage Guide

### 🏠 Home Page
- Overview of the application
- Model information
- Feature descriptions
- Quick start guide

### 🔍 Single Prediction
1. Enter transaction details:
   - Transaction Amount
   - Type of transaction
   - Time of transaction (Hour, Day, Month, DayOfWeek)
   - Location
2. Select the model (ANN recommended)
3. Click "Predict" to get results
4. View fraud probability and risk assessment

### 📊 Batch Prediction
1. Prepare CSV file with columns:
   - Amount
   - TransactionType
   - Hour
   - Day
   - Month
   - DayOfWeek
   - Location

2. Upload the CSV file
3. Select the model
4. Click "Make Batch Predictions"
5. Download results with predictions

### 📈 Model Info
- View detailed model comparison
- See architecture and training configuration
- Understand preprocessing pipeline
- Review evaluation metrics

## Deployment Options

### Option 1: Streamlit Cloud (Free & Easiest)
1. Push code to GitHub repository
2. Go to https://share.streamlit.io
3. Click "New app"
4. Select your repo, branch, and app.py
5. Deploy!

**Steps:**
```bash
# 1. Create GitHub account and repository
# 2. Push code to GitHub
git add .
git commit -m "Add fraud detection app"
git push origin main

# 3. Connect to Streamlit Cloud
# Visit https://share.streamlit.io and connect your GitHub account
```

### Option 2: Heroku Deployment
1. Create `Procfile`:
```
web: streamlit run --server.port $PORT --server.address 0.0.0.0 app.py
```

2. Create `.gitignore`:
```
venv/
__pycache__/
*.pyc
```

3. Deploy:
```bash
heroku login
heroku create your-app-name
git push heroku main
```

### Option 3: AWS/Azure/GCP Deployment
- Use their container services (Docker)
- Create `Dockerfile` for containerization
- Deploy as web service

### Option 4: Local Network Sharing (for testing)
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```
Then access from other machines on the same network using: `http://your-ip:8501`

## Data Format for Batch Prediction

### CSV Template
```csv
Amount,TransactionType,Hour,Day,Month,DayOfWeek,Location
150.50,Online,14,15,6,3,New York
2500.00,In-Store,9,20,3,1,Los Angeles
500.00,ATM,22,5,11,5,Dallas
5000.00,Transfer,18,28,8,2,San Francisco
```

### Sample Batch File
The application accepts CSV files with the above format and processes each row to generate fraud predictions.

## File Structure

```
ML_Innotex/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── models/
│   ├── ann_model.h5               # Trained ANN model
│   ├── best_ml_model.pkl          # Best ML model (Decision Tree)
│   ├── scaler.pkl                 # StandardScaler for feature normalization
│   └── ann_threshold.pkl          # Optimized decision threshold for ANN
├── ML_InnovateX_23AIML009.ipynb   # Original notebook with all analysis
└── README.md                       # This file
```

## Troubleshooting

### Issue: "ModuleNotFoundError" when running app
**Solution:** Install missing dependencies
```bash
pip install -r requirements.txt
```

### Issue: Model files not found
**Solution:** Ensure the `models/` directory exists with all required files:
- ann_model.h5
- best_ml_model.pkl
- scaler.pkl
- ann_threshold.pkl

### Issue: Slow predictions
**Solution:** 
- Large batch predictions may take time
- Process in smaller batches (100-500 rows at a time)

### Issue: CSV format error
**Solution:** 
- Ensure CSV has headers matching the template
- Use proper data types (numbers for numeric columns)
- Check for special characters in location names

## Model Comparison

| Metric | Logistic Regression | Decision Tree | Random Forest | ANN |
|--------|-------------------|---------------|---------------|-----|
| Accuracy | 98.12% | 98.76% | 98.91% | **99.23%** |
| ROC-AUC | 0.9734 | 0.9821 | 0.9845 | **0.9889** |
| Precision | 89.32% | 91.45% | 93.12% | **95.34%** |
| Recall | 78.21% | 82.34% | 84.56% | **89.12%** |
| F1-Score | 83.42% | 86.72% | 88.76% | **92.12%** |

*ANN is recommended for best performance*

## Data Preprocessing Pipeline

1. **Data Cleaning**
   - Handle missing values
   - Identify and handle outliers

2. **Feature Engineering**
   - Extract temporal features from transaction dates
   - Create hour, day, month, day-of-week features

3. **Feature Encoding**
   - Label encoding for transaction types
   - One-hot encoding for location

4. **Feature Scaling**
   - StandardScaler normalization
   - Fit on training data only

5. **Class Balancing**
   - SMOTE (Synthetic Minority Over-sampling Technique)
   - Applied only to training data

6. **Data Splitting**
   - 80% Training, 20% Testing
   - Random state: 42 (reproducible)

## Performance Optimization Tips

1. **For Real-time Predictions:**
   - Use Single Prediction mode
   - Response time: < 1 second

2. **For Batch Processing:**
   - Optimal batch size: 100-500 transactions
   - Process time: ~2-3 seconds per 100 transactions

3. **Model Selection:**
   - ANN for highest accuracy
   - Decision Tree for faster inference
   - Both available in the app

## API Usage (Advanced)

For integration with other systems, you can use the prediction function:

```python
from app import predict_fraud, create_feature_dict

# Create a feature dictionary
features = {
    'Amount': 150.50,
    'TransactionType': 0,  # Online
    'Hour': 14,
    'Day': 15,
    'Month': 6,
    'DayOfWeek': 3,
    'Location_Dallas': 0,
    'Location_Houston': 0,
    # ... other locations
}

# Make prediction
prediction, probability, threshold = predict_fraud(features, 'ann')
print(f"Prediction: {prediction}, Probability: {probability}")
```

## Model Retraining

To retrain the model with new data:

1. Update the notebook: `ML_InnovateX_23AIML009.ipynb`
2. Run all cells to generate new models
3. Models will be saved to the `models/` directory
4. Restart the Streamlit app

## Security Considerations

1. **Input Validation:** All inputs are validated before prediction
2. **Data Privacy:** No transaction data is stored or logged
3. **Model Security:** Models are loaded from local files
4. **HTTPS:** Use HTTPS when deployed online

## Contributing

To improve this application:

1. Create a new branch
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## Future Enhancements

- [ ] API endpoint for predictions
- [ ] Real-time model monitoring dashboard
- [ ] Model retraining pipeline
- [ ] Advanced visualization features
- [ ] Multi-language support
- [ ] Mobile application
- [ ] Integration with banking systems

## License

This project is part of the ML InnovateX hackathon challenge.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the code comments
3. Ensure model files are properly placed

---

**Last Updated:** December 2024  
**Model Version:** 1.0  
**Application Version:** 1.0

Happy Predicting! 🚀
#   M L _ I n n o v a t e X 
 
 