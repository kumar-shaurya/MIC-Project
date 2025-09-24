import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def preprocess_text(review):
    review = re.sub(r'<.*?>', '', review)
    review = re.sub(r'[^a-zA-Z\s]', '', review, re.I|re.A)
    review = review.lower()
    return review

dataset = 'Dataset.csv'
data_col = 'review'
pred_col = 'sentiment'

df = pd.read_csv(dataset)

df.dropna(subset=[data_col, pred_col], inplace=True)

df[data_col] = df[data_col].apply(preprocess_text)

X = df[data_col]
y = df[pred_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=7 #Thala for a reason
)

model_pipeline = Pipeline([('vectorizer', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),('classifier', LogisticRegression(random_state=7, max_iter=1000))])


print("Training dtaata")
model_pipeline.fit(X_train, y_train)
print("training complete")

y_pred = model_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100}%")
