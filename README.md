# 🌸 PCOSense - AI-Powered PCOS Health Assessment

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Model%20Accuracy-77.98%25-brightgreen.svg)]()

An intelligent web application that uses machine learning to predict PCOS (Polycystic Ovary Syndrome) stages and provides evidence-based, personalized lifestyle recommendations including yoga, nutrition, and exercise plans.

![PCOSense Demo](screenshot.png)

## 🎯 Overview

PCOSense combines clinical data analysis with machine learning to help women understand their PCOS risk and receive actionable health guidance. The system analyzes 9 key health indicators and provides:

- **PCOS Stage Prediction**: 5-stage classification (No PCOS → Stage 4 Severe)
- **Confidence Scoring**: Transparent ML predictions with confidence percentages
- **Risk Factor Analysis**: Identifies metabolic, hormonal, and lifestyle concerns
- **Personalized Recommendations**: Custom yoga routines, nutrition plans, and exercise programs

## ✨ Key Features

### 🤖 Machine Learning
- **Random Forest Classifier** trained on 541 clinical PCOS cases
- **77.98% accuracy** on test dataset
- 12 engineered features from medical indicators
- Balanced class weights for improved minority class prediction

### 🎨 User Interface
- Beautiful, responsive web design with smooth animations
- Real-time API status monitoring
- Mobile-friendly interface
- Intuitive form validation and error handling

### 💡 Intelligent Recommendations
Personalized suggestions based on:
- PCOS severity stage (0-4)
- Individual risk factors (BMI, cycle irregularity, hyperandrogenism)
- Metabolic indicators
- Evidence-based medical research

### 🏋️‍♀️ Recommendation Categories
- **Yoga**: Specific asanas, pranayama, and meditation techniques
- **Nutrition**: Anti-inflammatory foods, glycemic control, hormone-balancing nutrients
- **Exercise**: Cardio protocols, strength training, HIIT programs

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.7+
pip (Python package manager)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/PCOSense-AI-Health-Assessment.git
cd PCOSense-AI-Health-Assessment
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Start the API server**
```bash
python flask_api.py
```

4. **Open the web application**
```bash
open pcosense_ml.html
```
Or navigate to the file in your browser.

### One-Click Startup (Linux/Mac)
```bash
chmod +x start_pcosense.sh
./start_pcosense.sh
```

## 📊 Model Performance
```
Overall Accuracy: 77.98%
Training Samples: 432
Test Samples: 109

Class-wise Performance:
┌─────────────┬───────────┬────────┬──────────┐
│ PCOS Stage  │ Precision │ Recall │ F1-Score │
├─────────────┼───────────┼────────┼──────────┤
│ No PCOS     │   84%     │  89%   │   87%    │
│ Mild        │  100%     │  33%   │   50%    │
│ Moderate    │   29%     │  25%   │   27%    │
│ Significant │   67%     │  46%   │   55%    │
│ Severe      │   73%     │  92%   │   81%    │
└─────────────┴───────────┴────────┴──────────┘
```

### Feature Importance
1. **BMI** (14.0%)
2. **Weight** (11.9%)
3. **Hair Growth** (10.6%)
4. **Age** (10.5%)
5. **Weight Gain** (8.9%)

## 🏗️ Architecture
```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   Web Browser   │◄───────►│   Flask API     │◄───────►│  ML Model       │
│  (Frontend)     │  HTTP   │  (Backend)      │         │  (Random Forest)│
└─────────────────┘         └─────────────────┘         └─────────────────┘
```

**Tech Stack:**
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Backend**: Flask, Flask-CORS
- **ML**: scikit-learn, pandas, numpy
- **Model**: Random Forest Classifier (200 estimators)

## 📁 Project Structure
```
PCOSense/
├── pcosense_ml.html          # Web application interface
├── flask_api.py              # REST API server
├── train_pcos_model.py       # Model training script
├── pcos_model.pkl            # Trained ML model
├── scaler.pkl                # Feature scaler
├── model_metadata.json       # Model information
├── test_api.py               # API testing suite
├── start_pcosense.sh         # Startup script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── QUICKSTART.md             # Quick start guide
├── ARCHITECTURE.md           # System architecture
└── docs/                     # Additional documentation
```

## 🔧 API Documentation

### Health Check
```http
GET /health
```
**Response:**
>>>>>>> 326c782360a720470f9727bfc2c98fab165f85e3
```json
{
  "status": "healthy",
  "model": "Random Forest",
  "accuracy": 0.7798
}
```

### Prediction

```http
POST /predict
Content-Type: application/json
```
**Request Body:**
```json
{
  "age": 28,
  "bmi": 27.0,
  "cycle_length": 38,
  "cycle_regularity": 0,
  "hair_growth": 1,
  "pimples": 1,
  "hair_loss": 0,
  "weight_gain": 1,
  "skin_darkening": 1
}
```


**Response:**

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


## 🧪 Testing

Run the test suite:
```bash
python test_api.py
```

This validates:
- API health endpoint
- Prediction endpoint
- Multiple test scenarios
- Response format validation

## 📈 Dataset

The model was trained on a comprehensive PCOS dataset containing:
- **541 patients**
- **45 clinical features**
- Multiple diagnostic parameters including:
  - Hormonal markers (FSH, LH, AMH, testosterone)
  - Metabolic indicators (BMI, blood sugar, insulin resistance)
  - Physical symptoms (hirsutism, acne, hair loss)
  - Ultrasound findings (follicle counts)

## 🔒 Privacy & Security

- ✅ **Local Processing**: All data processed on your machine
- ✅ **No External APIs**: No data sent to third parties
- ✅ **No Data Storage**: Predictions not saved unless user chooses
- ✅ **CORS Enabled**: Secure cross-origin requests
- ✅ **Open Source**: Transparent, auditable code

## ⚠️ Medical Disclaimer

**Important**: PCOSense is an educational and informational tool. It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. 

- Always consult qualified healthcare providers regarding PCOS or any medical condition
- Use predictions as a starting point for discussions with your doctor
- Do not make medical decisions based solely on this tool
- Model accuracy of 78% means ~22% of predictions may be incorrect

## 🚀 Future Enhancements

- [ ] Integration with hormonal test results (FSH, LH, AMH)
- [ ] Ultrasound findings analysis
- [ ] Symptom tracking over time
- [ ] PDF report generation
- [ ] User authentication and history
- [ ] Mobile app (React Native)
- [ ] Multi-language support
- [ ] Wearable device integration

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/PCOSense-AI-Health-Assessment.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_api.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: This software is for educational and research purposes. Not approved for commercial medical use without proper certification and regulatory approval.

## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- Clinical PCOS dataset contributors
- scikit-learn development team
- Flask framework maintainers
- Medical research community for evidence-based guidelines

## 📚 References

- Rotterdam ESHRE/ASRM-Sponsored PCOS Consensus Workshop Group
- Evidence-based guidelines for PCOS management
- WHO guidelines on physical activity and nutrition

## 📧 Contact

For questions, suggestions, or collaboration:
- **Email**: your.email@example.com
- **LinkedIn**: [Your Profile](https://linkedin.com/in/yourprofile)
- **Twitter**: [@yourhandle](https://twitter.com/yourhandle)

## ⭐ Show Your Support

If you find this project helpful, please consider giving it a star ⭐ on GitHub!

---

**Disclaimer**: This is a research and educational project. Always consult healthcare professionals for medical advice.
```

---

## **Topics/Tags for GitHub:**
```
machine-learning
healthcare
pcos
womens-health
random-forest
flask-api
web-application
predictive-analytics
medical-ai
health-tech
python
scikit-learn
personalized-medicine
lifestyle-recommendations
clinical-decision-support
```

## **.gitignore file:**
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
dist/
*.egg-info/

# ML Models (optional - you might want to include these)
# *.pkl

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Environment variables
.env

# Test coverage
.coverage
htmlcov/

# Temporary files
tmp/
temp/
```

## **requirements.txt:**
```
flask==2.3.0
flask-cors==4.0.0
scikit-learn==1.3.0
pandas==2.0.0
numpy==1.24.0

