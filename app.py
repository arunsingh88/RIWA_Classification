from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os
import tensorflow as tf
import joblib


app = Flask(__name__)

# [Previous LSTMTicketClassifier class remains unchanged]
class LSTMTicketClassifier:
    def __init__(self):
        self.tokenizer = Tokenizer(num_words=10000)
        self.label_encoder = LabelEncoder()
        self.model = None
        self.max_length = 100  # Maximum length of sequences
        self.is_fitted = False
        
    def create_model(self, vocab_size, num_classes):
        """Create LSTM model architecture"""
        model = Sequential([
            Embedding(vocab_size, 128, input_length=self.max_length),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def train(self, data_path, epochs=10, batch_size=32):
        """Train the classifier using data from a CSV file"""
        # Load and prepare data
        df = pd.read_csv(data_path)
        
        # Encode labels
        self.label_encoder.fit(df['label'])
        encoded_labels = self.label_encoder.transform(df['label'])
        
        # Prepare text data
        self.tokenizer.fit_on_texts(df['text'])
        sequences = self.tokenizer.texts_to_sequences(df['text'])
        X = pad_sequences(sequences, maxlen=self.max_length)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, encoded_labels, test_size=0.2, random_state=42
        )
        
        # Create and train model
        vocab_size = len(self.tokenizer.word_index) + 1
        num_classes = len(self.label_encoder.classes_)
        
        self.model = self.create_model(vocab_size, num_classes)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        self.is_fitted = True
        
        # Evaluate model
        _, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return accuracy
    
    def classify_ticket(self,query):
        model = tf.keras.models.load_model(f"{MODEL_PATH}.h5")
        
        with open(f"{MODEL_PATH}.pickle", 'rb') as file:
            tokenizer = pickle.load(file)

        label_encoder_path = f"{MODEL_PATH}.joblib"
        label_encoder = joblib.load(label_encoder_path)

        # Get the classes and their corresponding encoded labels
    
        text = [query]
        text = tokenizer.texts_to_sequences(text)
        text = pad_sequences(text, maxlen=62)
        # Make predictions
        predictions = model.predict(text)
        predicted_label = []
        for i in np.argsort(predictions)[:,-5:][0][::-1]:
            if i == 24:
                continue
            predicted_label.append(label_encoder.inverse_transform([i])[0])
        predicted_label.append("Other template")
        return predicted_label
        
    def predict(self, text):
        """Predict ticket type with confidence scores"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Train the model first.")
            
        # Prepare input text
        sequences = self.tokenizer.texts_to_sequences([text])
        X = pad_sequences(sequences, maxlen=self.max_length)
        
        # Get predictions
        probabilities = self.model.predict(X)[0]
        
        # Create predictions list for items with confidence >= 0.40
        predictions = []
        for idx, prob in enumerate(probabilities):
            if prob >= 0.40:
                label = self.label_encoder.inverse_transform([idx])[0]
                predictions.append({
                    'label': label,
                    'confidence': float(prob)
                })
        
        # Sort by confidence score in descending order
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        return predictions
    
    def save_model(self, base_path):
        """Save the trained model and preprocessors"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Train the model before saving.")
            
        # Save Keras model
        self.model.save(f"{base_path}_model.h5")
        
        # Save tokenizer and label encoder
        with open(f"{base_path}_preprocessors.pkl", 'wb') as f:
            pickle.dump({
                'tokenizer': self.tokenizer,
                'label_encoder': self.label_encoder,
                'is_fitted': self.is_fitted
            }, f)
    
    @classmethod
    def load_model(cls, base_path):
        """Load a trained model and preprocessors"""
        if not os.path.exists(f"{base_path}_model"):
            return None
            
        model = cls()
        
        # Load Keras model
        model.model = load_model(f"{base_path}_model")
        
        # Load preprocessors
        with open(f"{base_path}_preprocessors.pkl", 'rb') as f:
            preprocessors = pickle.load(f)
            model.tokenizer = preprocessors['tokenizer']
            model.label_encoder = preprocessors['label_encoder']
            model.is_fitted = preprocessors['is_fitted']
            
        return model

# New Naive Bayes Classifier class
class NaiveBayesClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.label_encoder = LabelEncoder()
        self.model = MultinomialNB()
        self.is_fitted = False
    
    def train(self, data_path):
        """Train the Naive Bayes classifier using data from a CSV file"""
        # Load and prepare data
        df = pd.read_csv(data_path)
        
        # Encode labels
        self.label_encoder.fit(df['label'])
        encoded_labels = self.label_encoder.transform(df['label'])
        
        # Prepare text data
        X = self.vectorizer.fit_transform(df['text'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, encoded_labels, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Calculate accuracy
        accuracy = self.model.score(X_test, y_test)
        return accuracy
    

    
    def predict(self, text):
        """Predict ticket type with confidence scores"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Train the model first.")
        
        # Prepare input text
        X = self.vectorizer.transform([text])
        
        # Get predictions and probabilities
        probabilities = self.model.predict_proba(X)[0]
        
        # Create predictions list for items with confidence >= 0.40
        predictions = []
        for idx, prob in enumerate(probabilities):
            if prob >= 0.40:
                label = self.label_encoder.inverse_transform([idx])[0]
                predictions.append({
                    'label': label,
                    'confidence': float(prob)
                })
        
        # Sort by confidence score in descending order
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        return predictions
    
    def save_model(self, base_path):
        """Save the trained model and preprocessors"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Train the model before saving.")
        
        with open(f"{base_path}_nb_model.pkl", 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'label_encoder': self.label_encoder,
                'model': self.model,
                'is_fitted': self.is_fitted
            }, f)
    
    @classmethod
    def load_model(cls, base_path):
        """Load a trained model and preprocessors"""
        model_path = f"{base_path}_nb_model.pkl"
        if not os.path.exists(model_path):
            return None
        
        model = cls()
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
            model.vectorizer = saved_data['vectorizer']
            model.label_encoder = saved_data['label_encoder']
            model.model = saved_data['model']
            model.is_fitted = saved_data['is_fitted']
        
        return model

# Global instances
MODEL_PATH = 'lstm_ticket_classifier'
lstm_classifier = None
nb_classifier = None

def get_lstm_classifier():
    """Get or create LSTM classifier instance"""
    global lstm_classifier
    if lstm_classifier is None:
        lstm_classifier = LSTMTicketClassifier.load_model(MODEL_PATH)
        if lstm_classifier is None:
            lstm_classifier = LSTMTicketClassifier()
    return lstm_classifier

def get_nb_classifier():
    """Get or create Naive Bayes classifier instance"""
    global nb_classifier
    if nb_classifier is None:
        nb_classifier = NaiveBayesClassifier.load_model(MODEL_PATH)
        if nb_classifier is None:
            nb_classifier = NaiveBayesClassifier()
    return nb_classifier

# Existing endpoints
@app.route('/train', methods=['GET'])
def train():
    try:
        clf = get_lstm_classifier()
        accuracy = clf.train('classification.csv')
        clf.save_model(MODEL_PATH)
        
        return jsonify({
            'status': 'success',
            'message': 'LSTM Model trained successfully',
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
        if not request.is_json:
            return jsonify({
                'status': 'error',
                'message': 'Request must be JSON'
            }), 400
        
        data = request.get_json()
        if 'query' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Request must contain "query" field'
            }), 400
            
        query = data['query']
        
        predictions = LSTMTicketClassifier().classify_ticket(query)
        return jsonify(predictions)
        
        # return jsonify({
        #     'status': 'success',
        #     'predictions': predictions
        # }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# New Naive Bayes endpoints
@app.route('/train_nb', methods=['GET'])
def train_nb():
    try:
        clf = get_nb_classifier()
        accuracy = clf.train('classification.csv')
        clf.save_model(MODEL_PATH)
        
        return jsonify({
            'status': 'success',
            'message': 'Naive Bayes Model trained successfully',
            'accuracy': float(accuracy)
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/predict_nb', methods=['POST'])
def predict_nb():
    try:
        if not request.is_json:
            return jsonify({
                'status': 'error',
                'message': 'Request must be JSON'
            }), 400
        
        data = request.get_json()
        if 'query' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Request must contain "query" field'
            }), 400
            
        query = data['query']
        
        clf = get_nb_classifier()
        if not clf.is_fitted:
            return jsonify({
                'status': 'error',
                'message': 'Naive Bayes Model not trained. Please train the model first.'
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