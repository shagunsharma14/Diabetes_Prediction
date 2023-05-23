# Diabetes Prediction

This project aims to predict the likelihood of an individual having diabetes using machine learning techniques. It utilizes the PIMA Diabetes dataset, which contains various health-related features such as glucose level, blood pressure, BMI, etc.

## Table of Contents
- [Dependencies](#dependencies)
- [Data Collection and Analysis](#data-collection-and-analysis)
- [Data Standardization](#data-standardization)
- [Train Test Split](#train-test-split)
- [Training the Model](#training-the-model)
- [Model Evaluation](#model-evaluation)
- [Making a Predictive System](#making-a-predictive-system)

## Dependencies
The following dependencies are required for running the code:
- numpy
- pandas
- sklearn

## Data Collection and Analysis
The PIMA Diabetes dataset is loaded into a pandas DataFrame and analyzed to gain insights into the data. This includes printing the first few rows, obtaining the shape of the dataset, and exploring the statistical measures. The outcome column represents whether an individual is diabetic or non-diabetic.

```python
# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('/content/diabetes.csv') 

# printing the first 5 rows of the dataset
diabetes_dataset.head()

# number of rows and Columns in this dataset
diabetes_dataset.shape

# getting the statistical measures of the data
diabetes_dataset.describe()

# counting the number of instances for each outcome class
diabetes_dataset['Outcome'].value_counts()

# calculating mean values for each outcome class
diabetes_dataset.groupby('Outcome').mean()
```

## Data Standardization
To ensure that all features are on a similar scale, the data is standardized using the StandardScaler from sklearn. The standardized data is then assigned to the X variable, and the outcome labels are assigned to the Y variable.

```python
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data
Y = diabetes_dataset['Outcome']
```

## Train Test Split
The data is split into training and testing sets using the train_test_split function from sklearn. The test data size is set to 20% of the total data, and stratification is applied to maintain the distribution of outcome classes.

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
```

## Training the Model
A support vector machine (SVM) classifier with a linear kernel is chosen for training the model. The SVM classifier is fitted on the training data.

```python
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
```

## Model Evaluation
The accuracy score is calculated to evaluate the performance of the trained model. It is calculated separately for both the training and testing data.

```python
# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data:', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data:', test_data_accuracy)
```

## Making a Predictive System
A predictive system is implemented to make predictions on new input data. The input data, represented as a tuple, is converted to a numpy array, reshaped, and standardized using the trained scaler. Finally, the model predicts the outcome (diabetic or non-diabetic) based on the standardized input data.

```python
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

std_data = scaler.transform(input_data_reshaped)
prediction = classifier.predict(std_data)

if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')
```