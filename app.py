from flask import Flask, request, render_template
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
 
app = Flask(__name__, template_folder='template', static_folder='static')
 
# Load the trained SVM model and vectorizer
with open(r'C:\Users\Rawan\Desktop\Project1\model.pkl', 'rb') as f:
    model = pickle.load(f)

with open(r'C:\Users\Rawan\Desktop\Project1\vectorizer.pkl', 'rb') as g:
    vectorizer = pickle.load(g)
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text', False)
 
    # Preprocess the text
    def preprocess_text(text):
        text_clean = ''.join(e for e in text if e.isalnum() or e == ' ').lower()
        tokens = word_tokenize(text_clean)
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words]
        return ' '.join(tokens)
 
    processed_text = preprocess_text(text)
 
    # Generate features using the saved vectorizer
    X = vectorizer.transform([processed_text])
 
    # Make predictions
    prediction = model.predict(X)[0]
 
    # Return the prediction result to the user
    return render_template('index.html', text=text, prediction=prediction)
 

 
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)