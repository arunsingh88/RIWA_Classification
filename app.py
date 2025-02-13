from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import os

app = Flask(__name__)

class TicketClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.label_encoder = LabelEncoder()
        self.classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000))
        self.is_fitted = False
        
    def train(self, data_path):
        """Train the classifier using data from a CSV file"""
        df = pd.read_csv(data_path)
        
        self.label_encoder.fit(df['label'])
        encoded_labels = self.label_encoder.transform(df['label'])
        
        X = self.vectorizer.fit_transform(df['text'])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, encoded_labels, test_size=0.2, random_state=42
        )
        
        self.classifier.fit(X_train, y_train)
        self.is_fitted = True
        
        accuracy = self.classifier.score(X_test, y_test)
        return accuracy
        
    def predict(self, text):
        """Predict ticket type with confidence scores"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Train the model first.")
            
        X = self.vectorizer.transform([text])
        probabilities = self.classifier.predict_proba(X)[0]
        
        predictions = []
        for idx, prob in enumerate(probabilities):
            if prob >= 0.30:  # Only include predictions with confidence >= 0.30
                label = self.label_encoder.inverse_transform([idx])[0]
                predictions.append({
                    'label': label,
                    'confidence': float(prob)
                })
        
        # Sort by confidence score in descending order
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        return predictions
    
    def save_model(self, path):
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Train the model before saving.")
            
        model_data = {
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'classifier': self.classifier,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, path)
    
    @classmethod
    def load_model(cls, path):
        """Load a trained model"""
        if not os.path.exists(path):
            return None
            
        model = cls()
        model_data = joblib.load(path)
        
        model.vectorizer = model_data['vectorizer']
        model.label_encoder = model_data['label_encoder']
        model.classifier = model_data['classifier']
        model.is_fitted = model_data.get('is_fitted', True)
        return model

# Global classifier instance
MODEL_PATH = 'ticket_classifier.joblib'
classifier = None

def get_classifier():
    """Get or create classifier instance"""
    global classifier
    if classifier is None:
        # Try to load existing model
        classifier = TicketClassifier.load_model(MODEL_PATH)
        if classifier is None:
            # Create new instance if no saved model exists
            classifier = TicketClassifier()
    return classifier

@app.route('/train', methods=['GET'])
def train():
    try:
        clf = get_classifier()
        accuracy = clf.train('classification.csv')  # Make sure this file exists
        clf.save_model(MODEL_PATH)
        
        return jsonify({
            'status': 'success',
            'message': 'Model trained successfully',
            'accuracy': float(accuracy)
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if request contains JSON
        if not request.is_json:
            return jsonify({
                'status': 'error',
                'message': 'Request must be JSON'
            }), 400
        
        # Get query from request
        data = request.get_json()
        if 'query' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Request must contain "query" field'
            }), 400
            
        query = data['query']
        
        # Get classifier and make prediction
        clf = get_classifier()
        if not clf.is_fitted:
            return jsonify({
                'status': 'error',
                'message': 'Model not trained. Please train the model first.'
            }), 400
            
        predictions = clf.predict(query)
        
        return jsonify({
            'status': 'success',
            'predictions': predictions
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(port=8080)