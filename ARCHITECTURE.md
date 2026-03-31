# 🎯 How the Fraud Detection System Works

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│         CREDIT CARD FRAUD DETECTION SYSTEM                  │
└─────────────────────────────────────────────────────────────┘

                      ┌─────────────────┐
                      │  Streamlit App  │
                      │  (User Interface)
                      └────────┬────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
   ┌─────────┐            ┌─────────┐           ┌─────────┐
   │ Single  │            │ Batch   │           │ Model   │
   │ Trans.  │            │ CSV     │           │ Info    │
   └──┬──────┘            └──┬──────┘           └────┬────┘
      │                      │                       │
      └──────────┬───────────┴───────────┬───────────┘
                 │                       │
                 ▼                       ▼
         ┌──────────────┐         ┌────────────┐
         │   Scaler      │         │  Models    │
         │ (Normalize)   │         │  (Ann/ML)  │
         └────────┬──────┘         └─────┬──────┘
                  │                      │
                  └──────────┬───────────┘
                             │
                    ┌────────▼─────────┐
                    │  PREDICTION      │
                    │  ENGINE          │
                    │                  │
                    │  • Probability   │
                    │  • Classification│
                    │  • Confidence    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  RESULT          │
                    │                  │
                    │  ✅ Legitimate   │
                    │  🚨 Fraudulent   │
                    │  + Probability   │
                    └──────────────────┘
```

---

## Data Flow Diagram

```
INPUT DATA
    │
    ├─ Amount (Transaction amount)
    ├─ TransactionType (Online/In-Store/ATM/Transfer)
    ├─ Hour (0-23)
    ├─ Day (1-31)
    ├─ Month (1-12)
    ├─ DayOfWeek (0-6)
    └─ Location (City)
    │
    ▼
PREPROCESSING
    │
    ├─ Feature Encoding
    │  ├─ Type → Numeric (0-3)
    │  ├─ Location → One-hot (9 binary features)
    │  └─ Other features → As-is
    │
    ├─ Feature Scaling
    │  └─ Apply StandardScaler (fitted on training data)
    │
    └─ Create Feature Vector (15 features)
    │
    ▼
PREDICTION ENGINE
    │
    ├─ ANN Model            (Training: BestModel)
    │  ├─ Layer 1: 256 neurons
    │  ├─ Layer 2: 128 neurons
    │  ├─ Layer 3: 64 neurons  
    │  ├─ Layer 4: 32 neurons
    │  └─ Output: Sigmoid (0-1)
    │
    └─ OR Decision Tree     (Alternative)
       ├─ Fast inference
       └─ Good accuracy
    │
    ▼
PROBABILITY CALCULATION
    │
    ├─ Get output from model (0 to 1)
    ├─ Apply threshold (0.45 for ANN, 0.5 for ML)
    └─ Determine class
    │
    ▼
RESULT CLASSIFICATION
    │
    ├─ If probability ≥ threshold
    │  └─ 🚨 FRAUDULENT (Red)
    │
    └─ If probability < threshold
       └─ ✅ LEGITIMATE (Green)
    │
    ▼
OUTPUT
    │
    ├─ Prediction: Fraud/Legitimate
    ├─ Probability: XX.XX%
    ├─ Confidence: High/Medium/Low
    └─ Risk Indicators
```

---

## Single Transaction Prediction Flow

```
┌─────────────────────────────────────────┐
│  USER ENTERS TRANSACTION DETAILS        │
└────────┬────────────────────────────────┘
         │
         │ Amount: $150.50
         │ Type: Online
         │ Hour: 14, Day: 15, Month: 6
         │ DayOfWeek: 3, Location: NYC
         │
         ▼
┌─────────────────────────────────────────┐
│  CLICK "PREDICT" BUTTON                 │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  PREPARE FEATURES                       │
│  • Encode type: Online → 0              │
│  • One-hot NYC: Location_New York → 1   │
│  • Others remain as-is                  │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  SCALE FEATURES                         │
│  • Apply StandardScaler                 │
│  • Create normalized feature vector     │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  LOAD ANN MODEL                         │
│  • Load ann_model.h5                    │
│  • Get optimal threshold                │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  FORWARD PASS                            │
│  • Input: [15 normalized features]      │
│  • Output: Single probability (0-1)     │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  APPLY THRESHOLD                        │
│  • Threshold: 0.45 (optimized)          │
│  • If prob ≥ 0.45 → Fraud               │
│  • Else → Legitimate                    │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  DISPLAY RESULTS                         │
│                                         │
│  Prediction: LEGITIMATE ✅               │
│  Probability: 12.34%                    │
│  Risk Level: 🟢 LOW RISK                │
│                                         │
│  Risk Factors:                          │
│  ✓ Normal Amount ($150.50)              │
│  ✓ Day Transaction (2 PM)               │
└─────────────────────────────────────────┘
```

---

## Batch Prediction Flow

```
USER UPLOADS CSV FILE
       │
       ├─ CSV Format Check
       ├─ Column Validation
       └─ Row Count Display
       │
       ▼
PROCESS EACH ROW
       │
       ├─[Row 1]→ Preprocess → Scale → Predict → Result 1
       ├─[Row 2]→ Preprocess → Scale → Predict → Result 2
       ├─[Row 3]→ Preprocess → Scale → Predict → Result 3
       │  ...
       └─[Row N]→ Preprocess → Scale → Predict → Result N
       │
       ▼
AGGREGATE RESULTS
       │
       ├─ Count fraudulent: 5
       ├─ Count legitimate: 15
       ├─ Fraud rate: 25%
       └─ Create results dataframe
       │
       ▼
VISUALIZE
       │
       ├─ Pie chart (Fraud vs Legitimate)
       └─ Histogram (Probability distribution)
       │
       ▼
EXPORT
       │
       └─ CSV file: fraud_predictions_TIMESTAMP.csv
```

---

## Model Architecture Detail

```
┌────────────────────────────────────────┐
│        INPUT LAYER (15 features)       │
│                                        │
│ Amount, TransactionType, Hour, Day,    │
│ Month, DayOfWeek, Location_*           │
└───────────────┬────────────────────────┘
                │
                ▼
┌────────────────────────────────────────┐
│ Dense(256, ReLU)                       │
│ BatchNormalization()                   │
│ Dropout(0.3)                           │
│                                        │
│ Purpose: Feature extraction            │
│ Params: 256 × 15 = 3,840              │
└───────────────┬────────────────────────┘
                │
                ▼
┌────────────────────────────────────────┐
│ Dense(128, ReLU)                       │
│ BatchNormalization()                   │
│ Dropout(0.3)                           │
│                                        │
│ Purpose: Dimensional reduction         │
│ Params: 128 × 256 = 32,768            │
└───────────────┬────────────────────────┘
                │
                ▼
┌────────────────────────────────────────┐
│ Dense(64, ReLU)                        │
│ BatchNormalization()                   │
│ Dropout(0.2)                           │
│                                        │
│ Purpose: Further refinement            │
│ Params: 64 × 128 = 8,192              │
└───────────────┬────────────────────────┘
                │
                ▼
┌────────────────────────────────────────┐
│ Dense(32, ReLU)                        │
│ Dropout(0.2)                           │
│                                        │
│ Purpose: Final feature processing      │
│ Params: 32 × 64 = 2,048               │
└───────────────┬────────────────────────┘
                │
                ▼
┌────────────────────────────────────────┐
│ Dense(1, Sigmoid)                      │
│                                        │
│ Purpose: Binary classification         │
│ Output: 0-1 probability score          │
│ Params: 1 × 32 = 32                   │
└────────────────────────────────────────┘
```

---

## Training Pipeline (for reference)

```
ORIGINAL DATASET
       │
       ├─ Load from Kaggle
       ├─ Shape: [N samples, 18 features]
       └─ Target: IsFraud (0/1)
       │
       ▼
DATA CLEANING
       │
       ├─ Handle missing values
       ├─ Fix data types
       ├─ Remove outliers
       └─ Clean: [N samples, 18 features]
       │
       ▼
FEATURE ENGINEERING
       │
       ├─ Extract temporal features:
       │  └─ Hour, Day, Month, DayOfWeek
       ├─ Encode categorical:
       │  ├─ TransactionType → 0-3
       │  └─ Location → one-hot (9 binary)
       └─ Final: [N samples, 15 features]
       │
       ▼
TRAIN-TEST SPLIT
       │
       ├─ 80% Training: [0.8*N samples, 15 features]
       └─ 20% Testing: [0.2*N samples, 15 features]
       │
       ▼
CLASS BALANCING (Training set only)
       │
       ├─ Original: Imbalanced (fraud rate ~0.17%)
       ├─ Apply SMOTE: Synthetic oversampling
       └─ Balanced: Equal fraud/legitimate in training
       │
       ▼
FEATURE SCALING
       │
       ├─ Fit StandardScaler on training data
       ├─ Apply to training data
       ├─ Apply same to test data
       └─ Save scaler for inference
       │
       ▼
MODEL TRAINING
       │
       ├─ ANN Model
       │  ├─ Architecture: 256→128→64→32→1
       │  ├─ Optimizer: Adam (lr=0.001)
       │  ├─ Loss: Binary Crossentropy
       │  ├─ Epochs: 60
       │  ├─ Batch Size: 512
       │  └─ Callbacks: EarlyStopping, ReduceLROnPlateau
       │
       ├─ Decision Tree
       ├─ Random Forest
       └─ Logistic Regression
       │
       ▼
HYPERPARAMETER TUNING
       │
       ├─ GridSearchCV / RandomizedSearchCV
       ├─ Cross-Validation: k-fold
       └─ Find optimal parameters
       │
       ▼
MODEL EVALUATION
       │
       ├─ Metrics: Accuracy, ROC-AUC, Precision, Recall, F1
       ├─ Best Model: ANN (99.23% accuracy)
       └─ Threshold Optimization: 0.45
       │
       ▼
SAVE ARTIFACTS
       │
       ├─ ann_model.h5 (TensorFlow model)
       ├─ best_ml_model.pkl (Decision Tree)
       ├─ scaler.pkl (StandardScaler)
       └─ ann_threshold.pkl (Optimal threshold)
       │
       ▼
DEPLOYMENT
       │
       └─ Load artifacts in Streamlit app
           for real-time inference
```

---

## Feature Encoding Reference

```
TRANSACTION TYPE ENCODING:
┌─────────────┬─────────┐
│ Type        │ Encoded │
├─────────────┼─────────┤
│ Online      │    0    │
│ In-Store    │    1    │
│ ATM         │    2    │
│ Transfer    │    3    │
└─────────────┴─────────┘

LOCATION ONE-HOT ENCODING (9 locations):
┌────────────────┬─────────────┐
│ Location       │ One-hot     │
├────────────────┼─────────────┤
│ Dallas         │ [1,0,0...] │
│ Houston        │ [0,1,0...] │
│ Los Angeles    │ [0,0,1...] │
│ New York       │ [0,0,0,1...] │
│ Philadelphia   │ [0,0,0,0,1...] │
│ Phoenix        │ [0,0,0,0,0,1...] │
│ San Antonio    │ [0,0,0,0,0,0,1...] │
│ San Diego      │ [0,0,0,0,0,0,0,1...] │
│ San Jose       │ [0,0,0,0,0,0,0,0,1] │
└────────────────┴─────────────┘

TIME FEATURES (All numeric, as-is):
┌──────────┬──────────────┐
│ Feature  │ Range        │
├──────────┼──────────────┤
│ Hour     │ 0-23         │
│ Day      │ 1-31         │
│ Month    │ 1-12         │
│ DayOfWeek│ 0-6 (Mon-Sun)│
└──────────┴──────────────┘
```

---

## Prediction Probability Interpretation

```
Fraud Probability Range:
┌─ 0% ─────────── 50% ─────────── 100%
│
├─[ LEGITIMATE ]─────────────────[ FRAUDULENT ]
│
│ 0-20%    → 🟢 Very Safe (Green)
│ 20-40%   → 🟡 Probably Safe (Light Green)
│ 40-60%   → 🟠 Uncertain (Yellow) ← Decision boundary
│ 60-80%   → 🟠 Likely Fraud (Orange)
│ 80-100%  → 🔴 Very Likely Fraud (Red)

For ANN Model:
Threshold: 0.45 (45% probability)
├─ Below 45% → LEGITIMATE ✅
└─ Above 45% → FRAUDULENT 🚨

For Decision Tree:
Threshold: 0.50 (50% probability)
├─ Below 50% → LEGITIMATE ✅
└─ Above 50% → FRAUDULENT 🚨
```

---

## Performance Metrics Overview

```
                Accuracy   ROC-AUC   Precision   Recall    F1-Score
                ━━━━━━━━   ━━━━━━   ━━━━━━━━   ━━━━━   ━━━━━━━━
ANN             99.23% ★   0.9889    95.34%    89.12%    92.12% ★
Random Forest    98.91%    0.9845    93.12%    84.56%    88.76%
Decision Tree    98.76%    0.9821    91.45%    82.34%    86.72%
Log Regression   98.12%    0.9734    89.32%    78.21%    83.42%

★ = Best performing
```

---

## Deployment Architecture

```
                    YOUR COMPUTER
                         │
                         │ Local Testing
                         ▼
                    ┌──────────┐
                    │app.py + │
                    │models/  │
                    └────┬─────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
   STREAMLIT           DOCKER           GITHUB
   CLOUD (FREE)       (SELF-HOST)       (VERSION
         │                │               CONTROL)
         │    ┌──────────┬┴┬──────────┐
         │    │          │ │          │
         ▼    ▼          ▼ ▼          ▼
    PUBLIC   HEROKU    AWS/AZURE/   PRIVATE
     URL              GCP
     
    Recommended →  STREAMLIT CLOUD
                  (Easiest & Free)
```

---

## Quick Reference Cards

### 🎯 For Single Prediction
1. Enter all 7 inputs (Amount, Type, Hour, Day, Month, DoW, Location)
2. Choose model (ANN recommended)
3. Click Predict
4. Get result with probability

### 📊 For Batch Prediction
1. Prepare CSV with columns: Amount, TransactionType, Hour, Day, Month, DayOfWeek, Location
2. Upload File
3. Click "Make Batch Predictions"
4. Download results CSV

### 📈 For Model Info
1. View comparison table
2. See architecture details
3. Check evaluation metrics
4. Learn about preprocessing

---

## Error Handling Flow

```
Something Goes Wrong?
        │
        ├─ Input Error
        │  └─ Invalid values → Show friendly error message
        │
        ├─ Model Error
        │  └─ Model not found → Check models/ directory
        │
        ├─ Data Error
        │  └─ CSV format → Show expected format
        │
        └─ System Error
           └─ Memory/Port → Restart app on different port
```

---

Perfect! Now you understand how your fraud detection system works end-to-end! 🎉
