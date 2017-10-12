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
# Plot the data
######################################################

# Box and whisker plots
myData.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
# plt.savefig('box.png')
# plt.clf()

# Histogram
myData.hist()
# plt.savefig('hist.png')
# plt.clf()

# Scatterplots to look at 2 variables at once
# scatter plot matrix
scatter_matrix(myData)
# plt.savefig('scatter.png')
# plt.clf()

######################################################
# Evaluate algorithms
######################################################

# Separate training and final validation data set. First remove class
# label from data (X). Setup target class (Y)
# Then make the validation set 20% of the entire
# set of labeled data (X_validate, Y_validate)
valueArray = myData.values
X = valueArray[:,0:len(attributeNames)-1]		# Make it be size of feasures
Y = valueArray[:,len(attributeNames)-1]

# Best acc
best_acc = 0.0
model_name = ""
best_features = [-1,-1,-1,-1]

for a in range(0, 8):
	for b in range(0, 8):
		for c in range(0, 8):
			for d in range(0, 8):

				# Make sure the features are distinct
				if len(set([a,b,c,d])) < 4:
					continue

				# Choose 4 features
				X = valueArray[:, [a, b, c, d]]

				test_size = 0.20
				seed = 7
				X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=test_size, random_state=seed)

				######################################################
				# Normalization algorithms
				######################################################

				# Normalize features using scikit-learn
				X_train = preprocessing.normalize(X_train)
				X_validate = preprocessing.normalize(X_validate)


				# Setup 10-fold cross validation to estimate the accuracy of different models
				# Split data into 10 parts
				# Test options and evaluation metric
				num_folds = 10
				num_instances = len(X_train)
				seed = 7
				scoring = 'accuracy'

				######################################################
				# Use different algorithms to build models
				######################################################

				# Add each algorithm and its name to the model array
				models = []
				models.append(('KNN', KNeighborsClassifier()))
				models.append(('CART', DecisionTreeClassifier()))

				# Evaluate each model, add results to a results array,
				# Print the accuracy results (remember these are averages and std
				results = []
				names = []
				for name, model in models:
					kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=False)
					cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
					results.append(cv_results)
					names.append(name)
					msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
					print(msg)

					if cv_results.mean() > best_acc:
						best_acc = cv_results.mean()
						model_name = name
						best_features = [a,b,c,d]


				######################################################
				# For the best model (KNN), see how well it does on the
				# validation test
				######################################################
				# Make predictions on validation dataset
				# knn = KNeighborsClassifier()
				# knn.fit(X_train, Y_train)
				# predictions = knn.predict(X_validate)

				# print()
				# print('########## KNN: Test Results ##########')
				# print('Accuracy: ' + str(accuracy_score(Y_validate, predictions)))
				# print(confusion_matrix(Y_validate, predictions))
				# print(classification_report(Y_validate, predictions))

				# ######################################################
				# # For Decision Tree (CART), see how well it does on the
				# # validation test
				# ######################################################
				# # Make predictions on validation dataset
				# cart = DecisionTreeClassifier()
				# cart.fit(X_train, Y_train)
				# predictions = cart.predict(X_validate)

				# print()
				# print('########## Decision Tree: Test Results ##########')
				# print('Accuracy: ' + str(accuracy_score(Y_validate, predictions)))
				# print(confusion_matrix(Y_validate, predictions))
				# print(classification_report(Y_validate, predictions))


print("########## Best ACC ##########")
print("Best ACC: " + str(best_acc))
print("Best model: " + model_name)
print("Best feature index: " + str(best_features))
for i in best_features:
	print("#%f : %s" % (i+1, attributeNames[i]))