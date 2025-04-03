from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Example input data as a list of lists
X = [
    [25, 170, 60, 'red'],
    [30, 175, 70, 'blue'],
    [22, 160, 50, 'green'],
    [28, 180, 80, 'red']
]

# Target labels as a list
y = [0, 1, 0, 1]  # Binary labels: 0 and 1

# Extract the 'Category' column (4th column) for one-hot encoding
category_column = [[row[3]] for row in X]  # Reshape to 2D list for OneHotEncoder

# One-hot encode the 'Category' column
encoder = OneHotEncoder()
encoded_category = encoder.fit_transform(category_column).toarray()  # Convert to dense array

# Remove the original category column from X and append the encoded values
X_encoded = [row[:3] + list(encoded_category[i]) for i, row in enumerate(X)]

# Display the final input data for Random Forest
print("Encoded Input Features (X):")
for row in X_encoded:
    print(row)

# Initialize Random Forest model
rf_model = RandomForestClassifier()

# Fit the model
rf_model.fit(X_encoded, y)

# Display a message indicating successful training
print("\nRandom Forest model trained successfully!")
