from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras_preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load all three models and vectorizer
with open('lr_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load the Sequential model
seq_model = load_model('sqmodel.h5')

def clean_text(text):
    # Remove HTML tags
    text = re.sub('<.*?>', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def scrape_internshala(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
    }

    # Send a GET request to the URL
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code != 200:
        print("Failed to retrieve the page")
        return None

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the job description section
    # Internshala typically uses a class like 'internship_details' or similar for job content
    descriptions = soup.findAll('div', {'class': ['text-container','text-container additional_detail']})
    #print(description)

    if descriptions:
        # Extract and join all text from each div
        job_description = " ".join(
            [desc.get_text(separator=' ', strip=True) for desc in descriptions]
        )
        print(job_description[0:20])
        return job_description  # Return the job description as a single paragraph
    else:
        return None

def get_combined_prediction(description_vec, description_text):
    # Get Logistic Regression prediction probability
    lr_prob = lr_model.predict_proba(description_vec)[0][1]
    
    # Get SVM prediction probability (if using SVC, convert to probability)
    if hasattr(svm_model, 'predict_proba'):
        svm_prob = svm_model.predict_proba(description_vec)[0][1]
    else:
        # If SVM doesn't support probabilities, use decision function
        svm_decision = svm_model.decision_function(description_vec)[0]
        svm_prob = 1 / (1 + np.exp(-svm_decision))  # Convert to probability using sigmoid
    
    # Get Sequential model prediction
    # Note: You might need to adjust the input shape/preprocessing based on your sequential model
    #seq_input = description_vec.toarray()  # Convert sparse matrix to dense
    #seq_prob = seq_model.predict(seq_input)[0][0]  # Assuming binary classification
    
    tokenizer = pickle.load(open('tokenizer.pickle','rb'))
    print("Loaded tokernizer\n")
    #sample_sequence = tokenizer.texts_to_sequences(description_text)
    #print("Sample sequence tokenized\n")
    #padded_sample = pad_sequences(sample_sequence, maxlen=1500)
    #print("Sample padded")    
    
    # Step 2: Make the prediction
    #prediction = seq_model.predict(padded_sample)
    
    # Convert the text to sequence
    sequence = tokenizer.texts_to_sequences([description_text])
    
    # Pad the sequence
    padded_sequence = pad_sequences(sequence, maxlen=1500)
    
    # Get prediction
    seq_prob = seq_model.predict(padded_sequence)[0][0]
    
    # Interpret the result
    #seq_prob = (prediction > 0.5).astype(int)[0][0]
    print(lr_prob,svm_prob,seq_prob)
    
    # Calculate average probability
    avg_prob = (lr_prob + svm_prob + seq_prob) / 3
    
    # Get individual predictions
    predictions = {
        'logistic_regression': float(lr_prob),
        'svm': float(svm_prob),
        'sequential': float(seq_prob),
        'average': float(avg_prob)
    }
    
    return predictions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect_job', methods=['POST'])
def detect_job():
    try:
        data = request.get_json()
        job_url = data.get('url', '').strip()
        job_description = data.get('description', '').strip()

        # If URL is provided and description is empty, try to scrape
        if job_url and not job_description:
            if 'internshala.com' in job_url:
                job_description = scrape_internshala(job_url)
                if not job_description:
                    return jsonify({
                        'error': True,
                        'message': 'Unable to scrape job description from the provided URL'
                    })
            else:
                return jsonify({
                    'error': True,
                    'message': 'Only Internshala URLs are supported'
                })
        
        # If no valid input is provided
        if not job_description and not job_url:
            return jsonify({
                'error': True,
                'message': 'Please provide either a job description or a valid Internshala URL'
            })

        # Clean the job description
        cleaned_description = clean_text(job_description)
        print("Cleaned: ", cleaned_description[:20])
        
        # Transform the text using the vectorizer
        description_vec = vectorizer.transform([cleaned_description])

        # Get predictions from all models
        predictions = get_combined_prediction(description_vec, cleaned_description)
        
        # Determine if the job is fake based on average probability
        is_fake = predictions['average'] >= 0.5
        confidence = predictions['average'] * 100

        # Create detailed result message
        result_message = f"""
            This job posting appears to be {'FAKE' if is_fake else 'LEGITIMATE'}
            
            Confidence scores from each model:
            - Logistic Regression: {predictions['logistic_regression']*100:.2f}%
            - SVM: {predictions['svm']*100:.2f}%
            - Deep Learning: {predictions['sequential']*100:.2f}%
            
            Combined Confidence: {confidence:.2f}%
        """

        return jsonify({
            'error': False,
            'is_fake': is_fake,
            'message': result_message,
            'confidence': confidence,
            'detailed_predictions': predictions
        })

    except Exception as e:
        return jsonify({
            'error': True,
            'message': f'Error processing request: {str(e)}'
        })

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(debug=True, use_reloader=False)