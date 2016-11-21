#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.font_manager import FontProperties
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


from db import session, User, Business, Tip, userAndRespect


data = [[0]*16 for i in range(85901)]
data_without = [[0]*15 for i in range(85901)]
data_only = [[0] for i in range(85901)]
target = []

def trainSVMm():
    clf = svm.SVC(verbose = 2)
    clf.fit(data[:-17000],target[:-17000])	#save 20% of our data for the validation set.
    clf_only = svm.SVC(verbose = 2)
    clf_only.fit(data_only[:-17000],target[:-17000])	#save 20% of our data for the validation set.
    clf_without = svm.SVC(verbose = 2)
    clf_without.fit(data_without[:-17000],target[:-17000])
    joblib.dump(clf, 'models/svmModel.pkl') 		#save our model so we dont have to retrain every time
    joblib.dump(clf_only,'models/svmModelOnly.pkl')
    joblib.dump(clf_without, 'models/svmModelWithout.pkl')

def predictSVM():
    clf = joblib.load('models/svmModel.pkl')
    clf_without = joblib.load('models/svmModelWithout.pkl')
    clf_only = joblib.load('models/svmModelOnly.pkl')
    y_pred_without = clf_without.predict(data_without[-17000:])
    y_pred = clf.predict(data[-17000:])
    y_pred_only = clf_only.predict(data_only[-17000:])
    y_true = target[-17000:]
    acc_with = accuracy_score(y_true,y_pred)
    acc_without = accuracy_score(y_true,y_pred_without)
    acc_only = accuracy_score(y_true,y_pred_only)
    print("SVM - accuracy with social_network_feature : {0}".format(acc_with))
    print("SVM - accuracy without : {0}".format(acc_without))
    print("SVM - accuracy with only the social_network_feature : {0}".format(acc_only))
    return[acc_with,acc_without,acc_only]

def trainDecisionTree():
    clf = DecisionTreeClassifier()
    clf.fit(data[:-17000],target[:-17000])	
    clf_only = DecisionTreeClassifier()
    clf_only.fit(data_only[:-17000],target[:-17000])
    clf_without = DecisionTreeClassifier()
    clf_without.fit(data_without[:-17000],target[:-17000])
    joblib.dump(clf, 'models/treeModel.pkl') 
    joblib.dump(clf_only,'models/treeModelOnly.pkl')
    joblib.dump(clf_without, 'models/treeModelWithout.pkl')

def predictDecisionTree():
    clf = joblib.load('models/treeModel.pkl')
    clf_without = joblib.load('models/treeModelWithout.pkl')
    clf_only = joblib.load('models/treeModelOnly.pkl')
    y_pred_without = clf_without.predict(data_without[-17000:])
    y_pred = clf.predict(data[-17000:])
    y_pred_only = clf_only.predict(data_only[-17000:])
    y_true = target[-17000:]
    acc_with = accuracy_score(y_true,y_pred)
    acc_without = accuracy_score(y_true,y_pred_without)
    acc_only = accuracy_score(y_true,y_pred_only)
    print("Decision Tree - accuracy with social_network_feature : {0}".format(acc_with))
    print("Decision Tree - accuracy without : {0}".format(acc_without))
    print("Decision Tree - accuracy with only the social_network_feature : {0}".format(acc_only))
    return[acc_with,acc_without,acc_only]

def trainRandomForest():
    clf = RandomForestClassifier()
    clf.fit(data[:-17000],target[:-17000])	
    clf_only = RandomForestClassifier()
    clf_only.fit(data_only[:-17000],target[:-17000])
    clf_without = RandomForestClassifier()
    clf_without.fit(data_without[:-17000],target[:-17000])
    joblib.dump(clf, 'models/randForestModel.pkl') 
    joblib.dump(clf_only,'models/randForestModelOnly.pkl')
    joblib.dump(clf_without, 'models/randForestModelWithout.pkl')

def predictRandomForest():
    clf = joblib.load('models/randForestModel.pkl')
    clf_without = joblib.load('models/randForestModelWithout.pkl')
    clf_only = joblib.load('models/randForestModelOnly.pkl')
    y_pred_without = clf_without.predict(data_without[-17000:])
    y_pred = clf.predict(data[-17000:])
    y_pred_only = clf_only.predict(data_only[-17000:])
    y_true = target[-17000:]
    acc_with = accuracy_score(y_true,y_pred)
    acc_without = accuracy_score(y_true,y_pred_without)
    acc_only = accuracy_score(y_true,y_pred_only)
    print("RF Tree - accuracy with social_network_feature : {0}".format(acc_with))
    print("RF Tree - accuracy without : {0}".format(acc_without))
    print("RF Tree - accuracy with only the social_network_feature : {0}".format(acc_only))
    return[acc_with,acc_without,acc_only]

def trainMLP():
    clf = MLPClassifier()
    clf.fit(data[:-17000],target[:-17000])	
    clf_only = MLPClassifier()
    clf_only.fit(data_only[:-17000],target[:-17000])
    clf_without = MLPClassifier()
    clf_without.fit(data_without[:-17000],target[:-17000])
    joblib.dump(clf, 'models/mlpModel.pkl') 
    joblib.dump(clf_only,'models/mlpModelOnly.pkl')
    joblib.dump(clf_without, 'models/mlpModelWithout.pkl')

def predictMLP():
    clf = joblib.load('models/mlpModel.pkl')
    clf_without = joblib.load('models/mlpModelWithout.pkl')
    clf_only = joblib.load('models/mlpModelOnly.pkl')
    y_pred_without = clf_without.predict(data_without[-17000:])
    y_pred = clf.predict(data[-17000:])
    y_pred_only = clf_only.predict(data_only[-17000:])
    y_true = target[-17000:]
    acc_with = accuracy_score(y_true,y_pred)
    acc_without = accuracy_score(y_true,y_pred_without)
    acc_only = accuracy_score(y_true,y_pred_only)
    print("MLP - accuracy with social_network_feature : {0}".format(acc_with))
    print("MLP - accuracy without : {0}".format(acc_without))
    print("MLP - accuracy with only the social_network_feature : {0}".format(acc_only))
    return[acc_with,acc_without,acc_only]

def trainAda():
    clf = AdaBoostClassifier()
    clf.fit(data[:-17000],target[:-17000])	
    clf_only = AdaBoostClassifier()
    clf_only.fit(data_only[:-17000],target[:-17000])
    clf_without = AdaBoostClassifier()
    clf_without.fit(data_without[:-17000],target[:-17000])
    joblib.dump(clf, 'models/adaModel.pkl') 
    joblib.dump(clf_only,'models/adaModelOnly.pkl')
    joblib.dump(clf_without, 'models/adaModelWithout.pkl')

def predictAda():
    clf = joblib.load('models/adaModel.pkl')
    clf_without = joblib.load('models/adaModelWithout.pkl')
    clf_only = joblib.load('models/adaModelOnly.pkl')
    y_pred_without = clf_without.predict(data_without[-17000:])
    y_pred = clf.predict(data[-17000:])
    y_pred_only = clf_only.predict(data_only[-17000:])
    y_true = target[-17000:]
    acc_with = accuracy_score(y_true,y_pred)
    acc_without = accuracy_score(y_true,y_pred_without)
    acc_only = accuracy_score(y_true,y_pred_only)
    print("Ada - accuracy with social_network_feature : {0}".format(acc_with))
    print("Ada - accuracy without : {0}".format(acc_without))
    print("Ada - accuracy with only the social_network_feature : {0}".format(acc_only))
    return[acc_with,acc_without,acc_only]

def trainAll():
    trainSVM()
    trainDecisionTree()
    trainRandomForest()
    trainMLP()
    trainAda()


def plotBarGraph():

    print("Make sure all models are pretrained - try trainAll() if not")

    svm_data = predictSVM()
    dt_data = predictDecisionTree()
    RF_data = predictRandomForest()
    mlp_data = predictMLP()
    ada_data = predictAda()

    accuracys_with = [svm_data[0],dt_data[0],RF_data[0],mlp_data[0],ada_data[0]]
    accuracys_without = [svm_data[1],dt_data[1],RF_data[1],mlp_data[1],ada_data[1]]
    accuracys_only = [svm_data[2],dt_data[2],RF_data[2],mlp_data[2],ada_data[2]]
    
    labels = ['SVM','DecisionTree','RandomForest','MLP','AdaBoost']

    pos = list(range(len(accuracys_with)))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10,5))

    plt.bar(pos,accuracys_with,width,alpha=0.5,color='#EE3224')
    plt.bar([p + width for p in pos],accuracys_without,width,alpha=0.5,color='#F78F1E')
    plt.bar([p + width*2 for p in pos],accuracys_only,width,alpha=0.5,color='#FFC222')
    ax.set_ylabel('Classification Accuracy')

    ax.set_title('Classification Comparison')


    ax.set_xticks([p + 1.5 * width for p in pos])

    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(0,1.0,0.1))

    plt.xlim(min(pos)-width, max(pos)+width*4)
    plt.ylim([0, 1.0])

    fontP = FontProperties()
    fontP.set_size('small')
    plt.legend(['All features', 'Only word features', 'Only social network features'], loc='upper left', prop = fontP)
    fig.savefig('plot.pdf',bbox_inches='tight')
    plt.show()
    return ax

x = 0
for bus in session.query(Business):
    data[x] = [bus.Not,bus.good,bus.food,bus.place,bus.like,bus.out,bus.great,bus.very,bus.really,bus.about,bus.go,bus.time,bus.back,bus.service,bus.dont,bus.social_network_feature]
    data_without[x] =[bus.Not,bus.good,bus.food,bus.place,bus.like,bus.out,bus.great,bus.very,bus.really,bus.about,bus.go,bus.time,bus.back,bus.service,bus.dont]
    data_only[x] = [bus.social_network_feature]
    target.append(bus.success)
    x = x + 1


