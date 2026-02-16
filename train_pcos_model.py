import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading PCOS dataset...")
df = pd.read_csv('pcos_dataset.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names:")
print(df.columns.tolist())

# Display basic info
print(f"\nPCOS Distribution:")
print(df['PCOS (Y/N)'].value_counts())

# Clean column names
df.columns = df.columns.str.strip()

# Select relevant features for our web app inputs
# Mapping to our 10 input fields:
# 1. Age -> Age (yrs)
# 2. BMI -> BMI
# 3. Cycle length -> Cycle length(days)
# 4. Cycle regularity -> Cycle(R/I)
# 5. Hirsutism -> hair growth(Y/N)
# 6. Acne -> Pimples(Y/N)
# 7. Hair loss -> Hair loss(Y/N)
# 8. Weight gain -> Weight gain(Y/N)
# 9. Skin darkening -> Skin darkening (Y/N)
# 10. Mood changes -> We'll derive from other factors

# Create a copy with selected features
features_to_use = [
    'Age (yrs)', 
    'BMI', 
    'Cycle length(days)', 
    'Cycle(R/I)',
    'Weight gain(Y/N)',
    'hair growth(Y/N)',
    'Skin darkening (Y/N)',
    'Hair loss(Y/N)',
    'Pimples(Y/N)',
    'Pulse rate(bpm)',
    'Weight (Kg)',
    'RBS(mg/dl)',
    'PCOS (Y/N)'
]

# Create working dataframe
df_work = df[features_to_use].copy()

# Handle missing values
print("\nMissing values before cleaning:")
print(df_work.isnull().sum())

# Fill missing numeric values with median
numeric_cols = ['Age (yrs)', 'BMI', 'Cycle length(days)', 'Pulse rate(bpm)', 'Weight (Kg)', 'RBS(mg/dl)']
for col in numeric_cols:
    if df_work[col].isnull().sum() > 0:
        df_work[col].fillna(df_work[col].median(), inplace=True)

# Fill missing categorical with mode
categorical_cols = ['Cycle(R/I)', 'Weight gain(Y/N)', 'hair growth(Y/N)', 
                    'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)']
for col in categorical_cols:
    if df_work[col].isnull().sum() > 0:
        df_work[col].fillna(df_work[col].mode()[0], inplace=True)

print("\nMissing values after cleaning:")
print(df_work.isnull().sum())

# Convert categorical variables to numeric
# Cycle regularity: 2=Regular(R), 4=Irregular(I)
df_work['cycle_regularity_score'] = df_work['Cycle(R/I)'].map({2: 2, 4: 0}).fillna(1)

# Binary Y/N to 1/0
binary_cols = ['Weight gain(Y/N)', 'hair growth(Y/N)', 'Skin darkening (Y/N)', 
               'Hair loss(Y/N)', 'Pimples(Y/N)']
for col in binary_cols:
    df_work[col] = df_work[col].map({1: 1, 0: 0}).fillna(0)

# Create severity score for PCOS staging (0-4)
def calculate_pcos_severity(row):
    """Calculate PCOS severity based on multiple factors"""
    if row['PCOS (Y/N)'] == 0:
        return 0  # No PCOS
    
    score = 0
    # Cycle irregularity
    if row['Cycle(R/I)'] == 4:
        score += 1.5
    if row['Cycle length(days)'] > 35 or row['Cycle length(days)'] < 21:
        score += 1
    
    # Hyperandrogenism signs
    score += row['hair growth(Y/N)'] * 1.5
    score += row['Pimples(Y/N)'] * 0.5
    score += row['Hair loss(Y/N)'] * 0.5
    
    # Metabolic factors
    if row['BMI'] >= 30:
        score += 1.5
    elif row['BMI'] >= 25:
        score += 1
    
    score += row['Weight gain(Y/N)'] * 1
    score += row['Skin darkening (Y/N)'] * 1
    
    # Convert to stages
    if score <= 2:
        return 1  # Mild
    elif score <= 4:
        return 2  # Moderate
    elif score <= 6:
        return 3  # Significant
    else:
        return 4  # Severe

df_work['pcos_severity'] = df_work.apply(calculate_pcos_severity, axis=1)

print("\nPCOS Severity Distribution:")
print(df_work['pcos_severity'].value_counts().sort_index())

# Prepare features for model
X = df_work[[
    'Age (yrs)', 
    'BMI', 
    'Cycle length(days)', 
    'cycle_regularity_score',
    'Weight gain(Y/N)',
    'hair growth(Y/N)',
    'Skin darkening (Y/N)',
    'Hair loss(Y/N)',
    'Pimples(Y/N)',
    'Pulse rate(bpm)',
    'Weight (Kg)',
    'RBS(mg/dl)'
]].copy()

y = df_work['pcos_severity'].copy()

# Feature names for later use
feature_names = X.columns.tolist()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
print("\n" + "="*60)
print("Training Random Forest Classifier...")
print("="*60)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = rf_model.predict(X_train_scaled)
y_pred_test = rf_model.predict(X_test_scaled)

# Evaluate
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\nRandom Forest Results:")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_test, target_names=['No PCOS', 'Mild', 'Moderate', 'Significant', 'Severe']))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Train Gradient Boosting model for comparison
print("\n" + "="*60)
print("Training Gradient Boosting Classifier...")
print("="*60)

gb_model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

gb_model.fit(X_train_scaled, y_train)

y_pred_gb = gb_model.predict(X_test_scaled)
gb_accuracy = accuracy_score(y_test, y_pred_gb)

print(f"\nGradient Boosting Test Accuracy: {gb_accuracy:.4f}")

# Choose best model
if gb_accuracy > test_accuracy:
    best_model = gb_model
    model_name = "Gradient Boosting"
    best_accuracy = gb_accuracy
else:
    best_model = rf_model
    model_name = "Random Forest"
    best_accuracy = test_accuracy

print(f"\n{'='*60}")
print(f"Best Model: {model_name} with accuracy: {best_accuracy:.4f}")
print(f"{'='*60}")

# Save the model and scaler
print("\nSaving model and preprocessing objects...")
with open('pcos_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save feature names and model metadata
metadata = {
    'feature_names': feature_names,
    'model_type': model_name,
    'accuracy': float(best_accuracy),
    'feature_importance': feature_importance.to_dict('records'),
    'severity_mapping': {
        0: 'No PCOS',
        1: 'Stage 1 - Early/Mild',
        2: 'Stage 2 - Moderate',
        3: 'Stage 3 - Significant',
        4: 'Stage 4 - Severe'
    }
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\nModel saved successfully!")
print(f"- Model: pcos_model.pkl")
print(f"- Scaler: scaler.pkl")
print(f"- Metadata: model_metadata.json")

# Create sample prediction function
def predict_pcos_stage(age, bmi, cycle_length, cycle_regularity, 
                       weight_gain, hair_growth, skin_darkening, 
                       hair_loss, pimples, pulse_rate, weight, rbs):
    """
    Predict PCOS stage
    
    Parameters:
    - age: Age in years
    - bmi: Body Mass Index
    - cycle_length: Menstrual cycle length in days
    - cycle_regularity: 2 (Regular), 1 (Irregular), 0 (Very Irregular)
    - weight_gain: 1 (Yes), 0 (No)
    - hair_growth: 1 (Yes), 0 (No)
    - skin_darkening: 1 (Yes), 0 (No)
    - hair_loss: 1 (Yes), 0 (No)
    - pimples: 1 (Yes), 0 (No)
    - pulse_rate: Pulse rate in bpm
    - weight: Weight in kg
    - rbs: Random blood sugar in mg/dl
    """
    features = np.array([[age, bmi, cycle_length, cycle_regularity, 
                         weight_gain, hair_growth, skin_darkening, 
                         hair_loss, pimples, pulse_rate, weight, rbs]])
    
    features_scaled = scaler.transform(features)
    prediction = best_model.predict(features_scaled)[0]
    probabilities = best_model.predict_proba(features_scaled)[0]
    
    return {
        'stage': int(prediction),
        'stage_name': metadata['severity_mapping'][int(prediction)],
        'confidence': float(max(probabilities) * 100)
    }

# Test prediction
print("\n" + "="*60)
print("Testing sample prediction...")
print("="*60)

sample_prediction = predict_pcos_stage(
    age=28, bmi=27, cycle_length=38, cycle_regularity=0,
    weight_gain=1, hair_growth=1, skin_darkening=1,
    hair_loss=0, pimples=1, pulse_rate=75, weight=65, rbs=95
)

print(f"\nSample Prediction:")
print(f"Stage: {sample_prediction['stage']} - {sample_prediction['stage_name']}")
print(f"Confidence: {sample_prediction['confidence']:.2f}%")

print("\n" + "="*60)
print("Model training complete!")
print("="*60)
