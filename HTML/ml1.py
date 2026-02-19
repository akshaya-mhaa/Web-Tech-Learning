import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import classification_report

# Small labeled dataset
labeled_reviews = [
    "This product is amazing",
    "I hate this item",
    "Very good quality",
    "Worst experience ever"
]

labels = [1, 0, 1, 0]  # 1 = Positive, 0 = Negative

# Larger unlabeled dataset
unlabeled_reviews = [
    "Excellent service and fast delivery",
    "Not worth the money",
    "Highly recommended",
    "Terrible customer support",
    "Loved the packaging",
    "Very disappointing"
]

# Combine datasets
all_reviews = labeled_reviews + unlabeled_reviews
all_labels = labels + [-1] * len(unlabeled_reviews)

# Convert text to numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(all_reviews)

# Semi-supervised model
base_model = LogisticRegression()
model = SelfTrainingClassifier(base_model)

# Train model
model.fit(X, all_labels)

# Predict
predictions = model.predict(X)

print("Predictions:", predictions)