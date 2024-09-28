import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Download NLTK stopwords
nltk.download('stopwords')

# Sample dataset
data = {
    'resume_text': [
        "Dynamic software developer with extensive experience in building scalable applications.",
        "Detail-oriented data analyst with a passion for uncovering insights from data.",
        "Creative marketing specialist who thrives in a fast-paced environment.",
        "Team-oriented project manager with a proven track record in leading successful projects."
    ],
    'openness': [5, 4, 3, 4],
    'conscientiousness': [4, 5, 3, 5],
    'extraversion': [3, 2, 5, 4],
    'agreeableness': [4, 5, 4, 3],
    'emotional_stability': [3, 4, 4, 5]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Define a function to categorize personality traits into classes
def categorize_traits(row):
    if row['openness'] > 4:
        return 'High Openness'
    elif row['conscientiousness'] > 4:
        return 'High Conscientiousness'
    elif row['extraversion'] > 4:
        return 'High Extraversion'
    elif row['agreeableness'] > 4:
        return 'High Agreeableness'
    elif row['emotional_stability'] > 4:
        return 'High Emotional Stability'
    else:
        return 'Low Traits'

# Apply the function to create a target variable
df['personality_trait'] = df.apply(categorize_traits, axis=1)

# Prepare features and target variable
X = df['resume_text']
y = df['personality_trait']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for TF-IDF vectorization and model training
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stopwords.words('english'))),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Function to predict personality traits from a new resume
def predict_personality(resume):
    return pipeline.predict([resume])[0]

# Example usage
new_resume = "Highly motivated data scientist with expertise in machine learning and data visualization."
predicted_trait = predict_personality(new_resume)
print(f"The predicted personality trait for the new resume is: {predicted_trait}")
