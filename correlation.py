######################################################
# This code is a simple machine learning analysis
# using a well known, small easy to use data set about
# iris flowers. This lesson is adapted from
# Jason Brownlee's tutorials
#
# Machine learning task:
# Predict iris species from flower measurements
######################################################

# Import all your libraries.
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import preprocessing
import numpy as np

######################################################
# Load the data
######################################################
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
attributeNames = [	'pregnant-time',				# 1. Number of times pregnant
					'glucose-concentration',		# 2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
					'blood-pressure',				# 3. Diastolic blood pressure (mm Hg)
					'skin-fold-thickness',			# 4. Triceps skin fold thickness (mm)
					'serum-insulin',				# 5. 2-Hour serum insulin (mu U/ml)
					'body-mass-index',				# 6. Body mass index (weight in kg/(height in m)^2)
					'diabetes-pedigree-function',	# 7. Diabetes pedigree function
					'age',							# 8. Age (years)
					'class']						# 9. Class variable (0 or 1)
myData = pandas.read_csv(url, names=attributeNames)

######################################################
# Summarize the data
######################################################

# First few rows
print(myData.head(20))
print()

# Summary of data
print(myData.describe())
print()

# Look at the number of instances of each class
# class distribution
print(myData.groupby('class').size())


######################################################
# Check Correlation Coeeficient
######################################################

# Separate training and final validation data set. First remove class
# label from data (X). Setup target class (Y)
# Then make the validation set 20% of the entire
# set of labeled data (X_validate, Y_validate)
valueArray = myData.values
X = valueArray[:,0:len(attributeNames)-1]		# Make it be size of feasures
Y = valueArray[:,len(attributeNames)-1]

test_size = 0.20
seed = 7
X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=test_size, random_state=seed)


print("Size of X_train: " + str(len(X_train[:,[0]])))
print("Size of Y_train: " + str(len(Y_train)))
print("\n########## Correlation Coeficient ##########")
for i in range(0, X_train.shape[1]):
	coef = np.corrcoef(list(X_train[:,i]), list(Y_train))[0,1]
	print("Correlation Coeeficient of feature #" + str(i) + " : " + str(coef))



