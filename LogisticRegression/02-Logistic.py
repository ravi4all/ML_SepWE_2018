import random
import math
import csv

#Pregnancies
#Glucose
#BloodPressure
#SkinThickness
#Insulin
#BMI
#DiabetesPedigreeFunction
#Age
#Class

def load_dataset(filename):
    dataset = []
    with open(filename) as file:
        reader = csv.reader(file)
        for row in reader:
            dataset.append(row)
        
    return dataset

def str_to_float():
    pass

def minmax():
    minmax_data = []
    return minmax_data

# (x - min(x)) / (max(x) - min(x))
def normalize():
    pass

# k-fold cross validation
def cross_validation():
    pass

def prediction():
    x = coef[0]
    for i in range(len(row) - 1):
        x += coef[i+1] * row[i]
    return 1 / (1 + math.exp(-x))

def accuracy_metrics(actual, predicted):
    count = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            count += 1
    return count / len(actual) * 100

def evaluate_algorithm():
    pass

def sgd():
    pass

def logistic_regression():
    pass


path = 'pima-indians-diabetes.data.csv'
dataset = load_dataset(path)
