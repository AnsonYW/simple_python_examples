from sklearn import tree, svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import numpy as np

tree_clf = tree.DecisionTreeClassifier()
svm_clf = svm.SVC()
gaussian_clf = GaussianNB()
SGD_clf = SGDClassifier(loss="hinge")



# [height,weight,shoe_size]
x = [[181,80,44],[177,70,43],[160,60,38],[154,54,37],[166,65,40],
[190,90,47],[175,64,39],[177,70,40],[171,75,42],[181,85,43]]
# [gender]
y = ['male','male','female','female','male','male','female','female',
'female','male']
#train your model
tree_clf = tree_clf.fit(x,y)
svm_clf = svm_clf.fit(x,y)
gaussian_clf = gaussian_clf.fit(x,y)
SGD_clf = SGD_clf.fit(x,y)
#test your data
# prediction_tree = tree_clf.predict([[190,70,43]])
# prediction_svm = svm_clf.predict([[190,70,43]])
# prediction_gaussian = gaussian_clf.predict([[190,70,43]])
# prediction_SGD = SGD_clf.predict([[190,70,43]])

prediction_tree = tree_clf.predict(x)
prediction_svm = svm_clf.predict(x)
prediction_gaussian = gaussian_clf.predict(x)
prediction_SGD = SGD_clf.predict(x)

#calculate the accuracy score for each result
acc_tree = accuracy_score(y,prediction_tree) 
acc_svm = accuracy_score(y,prediction_svm)
acc_gaussian = accuracy_score(y,prediction_gaussian) 
acc_SGD = accuracy_score(y,prediction_SGD)

#print accuracy score
print(acc_tree,acc_svm,acc_gaussian, acc_SGD)
#print result
index = np.argmax([acc_tree,acc_svm,acc_gaussian, acc_SGD])
classifiers ={0:'Decision Tree',1:'SVM',2:'Gaussian Bayes',3:'SGDclassifier'}
print('The best gender classifer is {}'.format(classifiers[index]))
# print(prediction_tree,prediction_svm,prediction_gaussian,prediction_SGD)