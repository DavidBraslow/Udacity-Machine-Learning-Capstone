import numpy as np
print(np.__version__)
import pandas as pd
print(pd.__version__)
import graphviz as gv
print(gv.__version__)
import sklearn as sk
print(sk.__version__)

from sklearn.cross_validation import train_test_split
from time import time
from sklearn.metrics import f1_score
from sklearn import tree
from sklearn import dummy
from sklearn.preprocessing import Imputer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

student_data = pd.read_csv("Cleaned HSLS 100217.csv")
print("Student data read successfully!")

student_data.head(5)

n_students = len(student_data.index)
n_features = len(student_data.columns)-1

pstemp_sample_mask = student_data.isin({'S3FIELD_STEM': [1]})['S3FIELD_STEM']
pstemp_sample = student_data[pstemp_sample_mask]

n_pstemp = len(pstemp_sample)
print(n_pstemp)

n_failed = n_students - n_pstemp

pstemp_rate = n_pstemp*1.0 / n_students
print(pstemp_rate)

feature_cols = list(student_data.columns[:-1])
#print(feature_cols)
target_col = student_data.columns[-1] 

X_all = student_data[feature_cols]
y_all = student_data[target_col]

print("Feature values:")
print(X_all.head())

def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        
        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

X_all = preprocess_features(X_all)
print("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))

num_train = 3000
num_test = X_all.shape[0] - num_train

# TODO: Shuffle and split the dataset into the number of training and testing points above
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = num_test, random_state = 123)

print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print("Trained model in {:.4f} seconds".format(end - start))

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # Print and return results
    print("Made predictions in {:.4f} seconds.".format(end - start))
    return f1_score(target.values, y_pred, pos_label=0)


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    print("F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train)))
    print("F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test)))

# TODO: Initialize the model
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp = imp.fit(X_train)
X_train_imp = imp.transform(X_train)
X_test_imp = imp.transform(X_test)
X_all_imp = imp.transform(X_all)

clf = tree.DecisionTreeClassifier(random_state=123)
clf_comp = dummy.DummyClassifier(random_state=123)

# TODO: Set up the training set sizes
X_train_1000 = X_train_imp[:1000]
y_train_1000 = y_train[:1000]

X_train_2000 = X_train_imp[:2000]
y_train_2000 = y_train[:2000]

X_train_3000 = X_train_imp
y_train_3000 = y_train

# TODO: Execute the 'train_predict' function for each classifier and each training set size
#train_predict(clf, X_train_1000, y_train_1000, X_test_imp, y_test)
#train_predict(clf, X_train_2000, y_train_2000, X_test_imp, y_test)
train_predict(clf, X_train_3000, y_train_3000, X_test_imp, y_test)

#train_predict(clf_comp, X_train_1000, y_train_1000, X_test_imp, y_test)
#train_predict(clf_comp, X_train_2000, y_train_2000, X_test_imp, y_test)
train_predict(clf_comp, X_train_3000, y_train_3000, X_test_imp, y_test)

# TODO: Create the parameters list you wish to tune
parameters = {'max_depth':(3, 4, 5, 6, 7, 8, 9, 10), 'min_samples_leaf':(10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30)}
#parameters = {'max_depth':(3, 4, 5), 'min_samples_leaf':(2, 3, 4)}

# TODO: Make an f1 scoring function using 'make_scorer' 
f1_scorer = make_scorer(f1_score, pos_label=0)

# TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(clf, parameters,f1_scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_obj.fit(X_train_3000,y_train_3000)

# Get the estimator
clf = grid_obj.best_estimator_
print("Parameters: {}".format(clf.get_params()))

# Report the final F1 score for training and testing after parameter tuning
print("Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train_3000, y_train_3000)))
print("Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test_imp, y_test)))
    
# Use k-fold cross validation to evaluate model
scores = cross_val_score(clf, X_all_imp, y_all, cv=10, scoring=f1_scorer)    
print(scores)

feat_names = list(X_all.columns)

#Visualize tree
tree_data1 = tree.export_graphviz(clf, out_file='pstemp_tree.dot', feature_names=feat_names, class_names=['No PSTEMP','PSTEMP'], leaves_parallel = True, filled = True)
with open("pstemp_tree.dot") as f:
    dot_graph = f.read()
#print(dot_graph)
graph = gv.Source(source=dot_graph, filename='pstemp_tree', directory='C:/Users/david_000/Google Drive/HSLS_2009_v3_0_Stata_Datasets', format='png', engine='dot')

#graph = gv.Source(tree_data1, filename='pstemp_tree', directory='C:/Users/david_000/Google Drive/HSLS_2009_v3_0_Stata_Datasets', format='pdf') 
graph.render(filename='pstemp_tree', directory='C:/Users/david_000/Google Drive/HSLS_2009_v3_0_Stata_Datasets') 
#graph.render(filename='pstemp_tree', directory='C:/Users/david_000/Google Drive/HSLS_2009_v3_0_Stata_Datasets', view=True) 
    
    
