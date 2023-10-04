import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
data = pd.read_csv(r"C:\Users\rragh\Downloads\fake reviews dataset.csv")

# Preprocess the text
def preprocess_text(text):
    # Remove special characters and convert to lowercase
    text_clean = ''.join(e for e in text if e.isalnum() or e == ' ').lower()
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text_clean)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    
    return ' '.join(tokens)

data['review_clean'] = data['review'].apply(preprocess_text)

# Generate features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['review_clean'])
y = data['label']

#######

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate models
model = SVC(kernel='linear')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Training Accuracy : {:.3f}'.format(model.score(X_train, y_train)))
print('Test Accuracy : {:.3f}'.format(model.score(X_test, y_test))) 
print(classification_report(y_test, y_pred))

# Saving model to disk
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
