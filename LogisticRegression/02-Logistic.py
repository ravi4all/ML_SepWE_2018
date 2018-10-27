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

def str_to_float(dataset):
    for row in dataset:
        for i in range(len(row)):
            row[i] = float(row[i])

def minmax_data(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col = [row[i] for row in dataset]
        minmax.append([max(col),min(col)])
    return minmax

# (x - min(x)) / (max(x) - min(x))
def normalize(dataset, minmax):
    numer = 0
    denom = 0
    for row in dataset:
        for i in range(len(row)):
            numer = row[i] - minmax[i][1]
            denom = minmax[i][0] - minmax[i][1]
            row[i] = numer / denom

# k-fold cross validation
def cross_validation(dataset, n_folds):
    folds = []
    size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = []
        
        while len(fold) < size:
            index = random.randrange(len(dataset))
            fold.append(dataset.pop(index))
        
        folds.append(fold)
    return folds

def prediction(row, coef):
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

def evaluate_algorithm(dataset,learning_rate,nb_epoch,alogrithm,n_folds):
    folds = cross_validation(dataset, n_folds)
    scores = []
    for fold in folds:
        train = list(folds)
        train.remove(fold)
        train = sum(train,[])
        test = []
        for row in fold:
            row_copy = list(row)
            row_copy[-1] = None
            test.append(row_copy)
        
        actual = [row[-1] for row in fold]
        predicted = alogrithm(train, test,learning_rate,nb_epoch)
        score = accuracy_metrics(actual, predicted)
        scores.append(score)
    return scores
    

def sgd(dataset, nb_epoch, learning_rate):
    coef = [0] * len(dataset[0])
    for epoch in range(nb_epoch):
        for row in dataset:
            y_pred = prediction(row, coef)
            error = row[-1] - y_pred
            coef[0] = coef[0] + learning_rate * error * y_pred * (1 - y_pred)
            for i in range(len(row) - 1):
                coef[i+1] = coef[i+1] + learning_rate * error * y_pred * (1 - y_pred) * row[i]
    return coef

def logistic_regression(train, test,learning_rate,nb_epoch):
    coef = sgd(train, nb_epoch, learning_rate)
    pred = []
    for row in test:
        ypred = prediction(row, coef)
        ypred = round(ypred)
        pred.append(ypred)
    return pred

path = 'pima-indians-diabetes.data.csv'
dataset = load_dataset(path)
str_to_float(dataset)
minmax = minmax_data(dataset)
normalize(dataset, minmax)

n_folds = 5
learning_rate = 0.01
nb_epochs = 1000

#cross_validation(dataset,5)
acc = evaluate_algorithm(dataset,learning_rate,nb_epochs,logistic_regression,n_folds)
print(acc)