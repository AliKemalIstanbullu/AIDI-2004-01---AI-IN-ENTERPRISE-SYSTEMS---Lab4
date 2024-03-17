import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
fish_data = pd.read_csv('Fish.csv')

# Display summary statistics
print(fish_data.describe())

# Check for missing values
print(fish_data.isnull().sum())

# Visualize distributions of numerical features
plt.figure(figsize=(12, 8))
for i, column in enumerate(fish_data.columns[1:], 1):  # Skipping 'Species' for numerical data
    plt.subplot(2, 3, i)
    sns.histplot(fish_data[column], kde=True, bins=20)
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.show()

# Visualize the balance of fish species
plt.figure(figsize=(8, 6))
sns.countplot(data=fish_data, y='Species')
plt.title('Distribution of Fish Species')
plt.show()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
fish_data = pd.read_csv('Fish.csv')

# Preprocessing
# One-hot encode the 'Species' column as it is categorical
categorical_features = ['Species']
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot", 
                                  one_hot, 
                                  categorical_features)],
                                  remainder="passthrough")

# Define our features and target variable
X = fish_data.drop('Weight', axis=1)
y = fish_data['Weight']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection
# Create a pipeline that first transforms the data and then fits a model
model = Pipeline(steps=[('transformer', transformer),
                        ('model', LinearRegression())])

# Model Training
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")


from flask import Flask, request, render_template
app = Flask(__name__)
            
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = ""
    if request.method == 'POST':
        # Extract features from the form submission
        # Replace these with actual form field names and processing logic
        feature1 = request.form['feature1']
        # Process more features as needed

        # Convert features to the appropriate format for your model
        features = [float(feature1)]  # Update this as per your model's requirements

        # Load your model (ensure the model is accessible here)
        # model = load_model('your_model_path')

        # Make a prediction
        # prediction = model.predict([features])[0]  # Adjust depending on your model's output

        # Example placeholder prediction
        prediction = "123.45"  # Placeholder for actual prediction value

    return render_template('index.html', prediction=prediction)
