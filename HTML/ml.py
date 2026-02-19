import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Features (Square footage, Bedrooms, Location Score)
X = np.array([
    [1000, 2, 3],
    [1500, 3, 4],
    [800, 2, 2],
    [2000, 4, 5],
    [1200, 3, 3]
])
# Target (Price in Lakhs)
y = np.array([50, 75, 40, 100, 60])
# Model
model = LinearRegression()
model.fit(X, y)
# Predictions
predictions = model.predict(X)
# Output
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("Predicted Prices:", predictions)
print("Mean Squared Error:", mean_squared_error(y, predictions))