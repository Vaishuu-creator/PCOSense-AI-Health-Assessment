from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for web app

# Load model, scaler, and metadata
print("Loading trained model...")
with open('pcos_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Model loaded: {metadata['model_type']}")
print(f"Model accuracy: {metadata['accuracy']:.4f}")

def get_lifestyle_recommendations(stage, age, bmi, features):
    """Generate personalized lifestyle recommendations based on prediction and risk factors"""
    
    recommendations = {
        'yoga': [],
        'nutrition': [],
        'exercise': []
    }
    
    # Risk factor analysis
    high_bmi = bmi >= 25
    very_high_bmi = bmi >= 30
    irregular_cycle = features['cycle_regularity'] < 2
    hair_growth = features['hair_growth'] == 1
    weight_gain = features['weight_gain'] == 1
    skin_darkening = features['skin_darkening'] == 1
    pimples = features['pimples'] == 1
    hair_loss = features['hair_loss'] == 1
    
    # YOGA RECOMMENDATIONS
    base_yoga = [
        'Butterfly Pose (Baddha Konasana) - 5 minutes daily to improve pelvic circulation',
        'Child\'s Pose (Balasana) - 3-5 minutes for stress relief and hormonal balance',
        'Corpse Pose (Shavasana) - 10 minutes of deep relaxation'
    ]
    
    if stage >= 1:
        recommendations['yoga'].extend(base_yoga)
        recommendations['yoga'].append('Surya Namaskar (Sun Salutation) - Start with 5 rounds, build up to 12')
        recommendations['yoga'].append('Pranayama (Alternate Nostril Breathing) - 10-15 minutes daily')
    
    if stage >= 2 or irregular_cycle:
        recommendations['yoga'].append('Bhujangasana (Cobra Pose) - Stimulates ovaries and improves hormone production')
        recommendations['yoga'].append('Dhanurasana (Bow Pose) - Massages reproductive organs')
        recommendations['yoga'].append('Setu Bandhasana (Bridge Pose) - Balances hormones and reduces stress')
    
    if stage >= 3 or (hair_growth and pimples):
        recommendations['yoga'].append('Chakki Chalanasana (Mill Churning Pose) - Tones pelvic and abdominal organs')
        recommendations['yoga'].append('Padmasana (Lotus Pose) with meditation - 15-20 minutes for mental clarity')
        recommendations['yoga'].append('Viparita Karani (Legs-Up-the-Wall) - Improves blood flow to pelvic region')
    
    if stage == 0:
        recommendations['yoga'] = [
            'Maintain regular yoga practice 3-4 times per week',
            'Include basic Sun Salutations',
            'Practice stress-reducing poses like Child\'s Pose',
            'Add 10 minutes of meditation or deep breathing'
        ]
    
    # NUTRITION RECOMMENDATIONS
    base_nutrition = [
        'Increase fiber intake to 25-30g daily (vegetables, whole grains, legumes)',
        'Include omega-3 rich foods: flaxseeds, chia seeds, walnuts, fatty fish',
        'Stay hydrated - drink 8-10 glasses of water daily',
        'Reduce refined carbohydrates and added sugars'
    ]
    
    recommendations['nutrition'].extend(base_nutrition)
    
    if high_bmi or weight_gain:
        recommendations['nutrition'].extend([
            'Focus on low-glycemic foods: quinoa, sweet potato, brown rice, oats',
            'Practice portion control - use smaller plates',
            'Eat protein with every meal to stabilize blood sugar',
            'Avoid processed and fried foods'
        ])
    
    if very_high_bmi:
        recommendations['nutrition'].extend([
            'Consider time-restricted eating (12-hour eating window)',
            'Increase vegetable intake to half of each plate',
            'Limit fruit to 2 servings per day (lower sugar fruits preferred)'
        ])
    
    if hair_growth or pimples or skin_darkening:
        recommendations['nutrition'].extend([
            'Drink spearmint tea twice daily (helps reduce androgens)',
            'Include zinc-rich foods: pumpkin seeds, chickpeas, lentils',
            'Add anti-inflammatory spices: turmeric, ginger, cinnamon',
            'Consider inositol supplementation (consult healthcare provider)'
        ])
    
    if stage >= 2:
        recommendations['nutrition'].extend([
            'Add cinnamon to meals for insulin sensitivity (1/2 - 1 tsp daily)',
            'Include foods rich in magnesium: spinach, almonds, dark chocolate',
            'Consider vitamin D supplementation if levels are low',
            'Limit dairy products, especially conventional dairy'
        ])
    
    if irregular_cycle:
        recommendations['nutrition'].extend([
            'Include vitamin B-rich foods: eggs, leafy greens, whole grains',
            'Add healthy fats: avocado, olive oil, nuts',
            'Consider evening primrose oil (consult healthcare provider)'
        ])
    
    if stage == 0:
        recommendations['nutrition'] = [
            'Maintain a balanced Mediterranean-style diet',
            'Include plenty of colorful vegetables and fruits',
            'Choose whole grains over refined grains',
            'Stay hydrated and limit processed foods',
            'Include omega-3 sources 2-3 times per week'
        ]
    
    # EXERCISE RECOMMENDATIONS
    base_exercise = [
        'Brisk walking - 30 minutes, 5 days per week',
        'Strength training - 2-3 times per week (all major muscle groups)',
        'Include rest days for recovery'
    ]
    
    recommendations['exercise'].extend(base_exercise)
    
    if high_bmi or weight_gain:
        recommendations['exercise'].extend([
            'Add HIIT workouts - 20-25 minutes, 2-3 times per week',
            'Increase daily step count to 10,000+ steps',
            'Swimming or water aerobics - excellent low-impact option',
            'Cycling - 30-45 minutes, 3 times per week'
        ])
    
    if stage >= 2:
        recommendations['exercise'].extend([
            'Consistent exercise is crucial - aim for 150-200 minutes per week',
            'Mix cardio and resistance training',
            'Pelvic floor exercises (Kegels) - daily',
            'Consider working with a fitness trainer familiar with PCOS'
        ])
    
    if stage >= 3:
        recommendations['exercise'].extend([
            'Don\'t over-exercise - excessive cardio can worsen hormonal imbalance',
            'Focus on compound movements: squats, deadlifts, push-ups',
            'Include mobility and flexibility work',
            'Track your cycle and adjust intensity accordingly'
        ])
    
    recommendations['exercise'].extend([
        'Stress management: meditation, deep breathing, or tai chi - 10-15 minutes daily',
        'Prioritize sleep - aim for 7-9 hours per night',
        'Limit high-intensity exercise if experiencing high stress'
    ])
    
    if stage == 0:
        recommendations['exercise'] = [
            'Maintain regular physical activity - 150 minutes per week',
            'Mix cardio (walking, jogging, cycling) with strength training',
            'Stay active throughout the day - take movement breaks',
            'Include activities you enjoy for long-term adherence',
            'Practice stress-reduction techniques regularly'
        ]
    
    return recommendations

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for PCOS prediction"""
    try:
        data = request.json
        
        # Extract features from request
        age = float(data.get('age', 28))
        bmi = float(data.get('bmi', 22))
        cycle_length = int(data.get('cycle_length', 28))
        cycle_regularity = int(data.get('cycle_regularity', 2))
        weight_gain = int(data.get('weight_gain', 0))
        hair_growth = int(data.get('hair_growth', 0))
        skin_darkening = int(data.get('skin_darkening', 0))
        hair_loss = int(data.get('hair_loss', 0))
        pimples = int(data.get('pimples', 0))
        
        # Calculate additional features
        # Estimate pulse rate (average for age and BMI)
        pulse_rate = 72 + (bmi - 22) * 0.5 if bmi > 22 else 72
        
        # Estimate weight from BMI (assuming average height of 160cm)
        weight = bmi * 2.56  # (160/100)^2 * bmi
        
        # Estimate RBS (random blood sugar)
        rbs = 90 + (bmi - 22) * 2 if bmi > 22 else 90
        
        # Prepare feature array
        features_array = np.array([[
            age, bmi, cycle_length, cycle_regularity,
            weight_gain, hair_growth, skin_darkening,
            hair_loss, pimples, pulse_rate, weight, rbs
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = int(model.predict(features_scaled)[0])
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = float(max(probabilities) * 100)
        
        # Get stage information
        stage_name = metadata['severity_mapping'][str(prediction)]
        
        # Generate recommendations
        feature_dict = {
            'cycle_regularity': cycle_regularity,
            'hair_growth': hair_growth,
            'weight_gain': weight_gain,
            'skin_darkening': skin_darkening,
            'pimples': pimples,
            'hair_loss': hair_loss
        }
        
        recommendations = get_lifestyle_recommendations(prediction, age, bmi, feature_dict)
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'stage': prediction,
                'stage_name': stage_name,
                'confidence': round(confidence, 2)
            },
            'recommendations': recommendations,
            'risk_factors': {
                'high_bmi': bmi >= 25,
                'irregular_cycles': cycle_regularity < 2,
                'hyperandrogenism': hair_growth == 1 or pimples == 1,
                'metabolic_issues': skin_darkening == 1 or bmi >= 30
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': metadata['model_type'],
        'accuracy': metadata['accuracy']
    })

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'PCOSense ML API',
        'version': '1.0',
        'model': metadata['model_type'],
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)'
        }
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("PCOSense ML API Server")
    print("="*60)
    print(f"Model: {metadata['model_type']}")
    print(f"Accuracy: {metadata['accuracy']:.2%}")
    print(f"Server starting on http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
