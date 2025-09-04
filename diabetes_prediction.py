## Importing Libraries
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

"""## Data Collection and Analysis for the PIMA Diabetes Dataset"""

# -> Load the diabetes dataset from CSV file and display the first 5 rows
diabetes = pd.read_csv('/content/diabetes.csv')
diabetes.head()

# -> Display the number of rows and columns in the dataset
diabetes.shape

# -> Generate descriptive statistics (count, mean, std, min, quartiles, max) for numerical columns
diabetes.describe()

# -> Count the number of occurrences for each class (0 = non-diabetic, 1 = diabetic)
diabetes['Outcome'].value_counts()

"""0 --> Non Diabetes

1 --> Diabetes
"""

# -> Group the dataset by Outcome and calculate the mean of each feature
diabetes.groupby('Outcome').mean()

# -> Separate the features (X) and the target variable (Y)
X = diabetes.drop(columns='Outcome', axis=1)
Y = diabetes['Outcome']

print(X)

print(Y)

"""## Data StandardScaler"""

# -> Initialize the StandardScaler and fit it to the feature data (X) to learn mean and std for scaling
scaler = StandardScaler()
scaler.fit(X)

# -> Transform the feature data (X) using the fitted scaler to get standardized values (mean=0, std=1)
Standardized_data = scaler.transform(X)

print(Standardized_data)

# -> Update X with the standardized feature data and keep Y as the target variable (Outcome)
X = Standardized_data
Y = diabetes['Outcome']

print(X)

print(Y)

"""## Training, testing, and splitting data."""

X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state = 2)

print(X.shape, X_train.shape, X_test.shape)

"""## Training the Model"""

# -> Initialize an SVM (Support Vector Machine) classifier with a linear kernel
classifier = svm.SVC(kernel = 'linear')

# -> Train the SVM classifier on the training data (X_train, Y_train)
classifier.fit(X_train, Y_train)

"""## Model Evaluation

## Accuracy Score
"""

# -> Predict the labels of the training data
X_train_prediction = classifier.predict(X_train)
# -> Calculate accuracy score by comparing predictions with actual labels
training_accuracy = accuracy_score(X_train_prediction, Y_train)
print(f"Accuracy score of the training data {training_accuracy}")

# -> Predict the labels of the test data
X_test_prediction = classifier.predict(X_test)
# -> Calculate accuracy score by comparing predictions with test labels
test_accuracy = accuracy_score(X_test_prediction, Y_test)
print(f"Accuracy score of the test data {test_accuracy}")

"""## Making a Predictive System"""

predictive_data = (9,171,110,24,240,45.4,0.721,54)
# -> Convert the input tuple to a NumPy array
predictive_array = np.asarray(predictive_data)
# -> Reshape the array to 2D (1 row, 8 columns) to match model input format
predictive_reshaped = predictive_array.reshape(1,-1)
# -> Standardize the input data using the same scaler fitted on training data
std_data = scaler.transform(predictive_reshaped)
print(std_data)
# -> Make prediction using the trained classifier (0 = non-diabetic, 1 = diabetic)
prediction = classifier.predict(std_data)

print(prediction)
# -> Display a human-readable message based on the predicti
if prediction[0] == 0:
  print('The person is not diabetes')
else:
  print('The person is diabetes')

