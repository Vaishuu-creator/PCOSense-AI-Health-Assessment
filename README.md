# PCOSense - AI-Powered PCOS Health Assessment System

An intelligent web application that uses machine learning to predict PCOS stages and provide personalized lifestyle recommendations.

## 🎯 Features

- **Machine Learning Prediction**: Random Forest model trained on 541 clinical PCOS cases
- **78% Accuracy**: Reliable predictions based on real medical data
- **5-Stage Classification**: From No PCOS to Stage 4 (Severe)
- **Personalized Recommendations**: Custom yoga, nutrition, and exercise plans based on:
  - PCOS severity stage
  - Individual risk factors
  - BMI and metabolic indicators
  - Hyperandrogenism symptoms
- **Beautiful UI**: Elegant, responsive design with smooth animations
- **Real-time API**: Flask-based REST API for predictions

## 📊 Model Details

- **Algorithm**: Random Forest Classifier
- **Training Data**: 541 patients, 45 clinical features
- **Test Accuracy**: 77.98%
- **Features Used**: 
  - Age, BMI
  - Cycle length and regularity
  - Hyperandrogenism indicators (hair growth, acne, hair loss)
  - Metabolic markers (weight gain, skin darkening)
  - Physiological parameters (pulse rate, blood sugar)

## 🚀 Quick Start

### Prerequisites

```bash
pip install flask flask-cors scikit-learn pandas numpy pickle5
```

### Step 1: Start the Flask API Server

```bash

python flask_api.py
```

You should see:
```
============================================================
PCOSense ML API Server
============================================================
Model: Random Forest
Accuracy: 77.98%
Server starting on http://localhost:5000
============================================================
```

### Step 2: Open the Web Application

Open `pcosense_ml.html` in your web browser. The API status indicator in the top-right should show "AI Model Online".

### Step 3: Complete the Assessment

Fill in the 9 health parameters:
1. Age (years)
2. BMI (Body Mass Index)
3. Menstrual cycle length (days)
4. Cycle regularity
5. Excess hair growth (hirsutism)
6. Acne/pimples
7. Hair loss/thinning
8. Recent weight gain
9. Skin darkening (acanthosis nigricans)

Click "Analyze with AI" to get your results!

## 📁 File Structure

```
/PCOsense/
├── pcos_model.pkl           # Trained Random Forest model
├── scaler.pkl               # Feature scaler for normalization
├── model_metadata.json      # Model information and feature importance
├── train_pcos_model.py      # Training script
├── flask_api.py             # REST API server
├── pcosense_ml.html         # Web application (ML-powered)
└── README.md               # This file
```

## 🔧 API Endpoints

### Health Check
```bash
GET http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "model": "Random Forest",
  "accuracy": 0.7798
}
```

### Prediction
```bash
POST http://localhost:5000/predict
Content-Type: application/json

{
  "age": 28,
  "bmi": 27,
  "cycle_length": 38,
  "cycle_regularity": 0,
  "hair_growth": 1,
  "pimples": 1,
  "hair_loss": 0,
  "weight_gain": 1,
  "skin_darkening": 1
}
```

Response:
```json
{
  "success": true,
  "prediction": {
    "stage": 3,
    "stage_name": "Stage 3 - Significant",
    "confidence": 72.5
  },
  "recommendations": {
    "yoga": [...],
    "nutrition": [...],
    "exercise": [...]
  },
  "risk_factors": {
    "high_bmi": true,
    "irregular_cycles": true,
    "hyperandrogenism": true,
    "metabolic_issues": true
  }
}
```

## 🎨 PCOS Stages

### Stage 0 - No PCOS
- Low likelihood of PCOS
- Hormonal health appears balanced
- Preventive lifestyle recommendations

### Stage 1 - Early/Mild
- Mild PCOS indicators detected
- Early intervention highly effective
- Focus on lifestyle optimization

### Stage 2 - Moderate
- Moderate PCOS symptoms present
- Lifestyle modifications important
- May benefit from medical consultation

### Stage 3 - Significant
- Significant PCOS indicators
- Medical consultation recommended
- Comprehensive lifestyle changes needed

### Stage 4 - Severe
- Severe PCOS symptoms detected
- Immediate medical attention advised
- Intensive lifestyle intervention required

## 🧘‍♀️ Recommendation Categories

### Yoga
- Asanas for hormonal balance
- Stress-relief poses
- Pranayama (breathing exercises)
- Stage-specific intensity levels

### Nutrition
- Anti-inflammatory foods
- Blood sugar management
- Hormone-balancing nutrients
- Personalized dietary strategies

### Exercise
- Cardio recommendations
- Strength training protocols
- HIIT programs for weight management
- Recovery and stress management

## 🔬 Model Training

To retrain the model with updated data:

```bash
python train_pcos_model.py
```

This will:
1. Load and clean the PCOS dataset
2. Engineer features and severity scores
3. Train Random Forest and Gradient Boosting models
4. Compare models and save the best one
5. Generate feature importance analysis
6. Save model artifacts (model, scaler, metadata)

## 📈 Feature Importance

Top features for PCOS prediction:
1. BMI (14.0%)
2. Weight (11.9%)
3. Hair growth (10.6%)
4. Age (10.5%)
5. Weight gain (8.9%)
6. Cycle regularity (8.5%)
7. Blood sugar (7.8%)
8. Skin darkening (6.4%)

## ⚕️ Medical Disclaimer

This application is for informational and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers regarding PCOS or any medical condition.

## 🛠️ Troubleshooting

### API Status shows "Offline"
- Ensure Flask server is running: `python flask_api.py`
- Check that port 5000 is not blocked
- Verify all dependencies are installed

### CORS Errors
- Make sure flask-cors is installed: `pip install flask-cors`
- Restart the Flask server

### Low Confidence Predictions
- Ensure all input fields are filled accurately
- Extreme or unusual values may reduce confidence
- Model performs best with typical ranges

## 📊 Performance Metrics

```
Overall Accuracy: 77.98%
Training Accuracy: 95.83%

Per-Class Performance:
- No PCOS (Stage 0): 87% F1-score
- Mild (Stage 1): 50% F1-score
- Moderate (Stage 2): 27% F1-score
- Significant (Stage 3): 55% F1-score
- Severe (Stage 4): 81% F1-score
```

## 🚀 Future Enhancements

- [ ] Add support for hormonal test results (FSH, LH, AMH)
- [ ] Integrate ultrasound findings (follicle counts)
- [ ] Track symptoms over time
- [ ] Add export functionality for reports
- [ ] Implement user authentication and history
- [ ] Mobile app version
- [ ] Integration with wearable devices

## 📝 Dataset

The model was trained on a comprehensive PCOS dataset containing:
- 541 patients
- 45 clinical features
- Multiple diagnostic parameters
- Hormonal, metabolic, and physical indicators

## 👩‍⚕️ Credits

Developed using clinical PCOS data and evidence-based medical research. Machine learning model trained on peer-reviewed diagnostic criteria.

## 📄 License

Educational and research purposes. Not for commercial medical use without proper certification and regulatory approval.

---

**Version**: 1.0  
**Model**: Random Forest Classifier  
**Last Updated**: February 2026