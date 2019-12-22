import numpy as np
import sklearn.datasets
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import ssl
ssl._create_default_https_context = ssl._create_unverified_context



# EX1

def knn(x_train, y_train):
    k = np.arange(20) + 1
    weight_options = ['uniform', 'distance']
    param_grid = dict(n_neighbors=k, weights=weight_options)

    results = grid_search(KNeighborsClassifier(), param_grid, x_train, y_train)
    # presResul conte la precisio dels dos weights ordenat per k-nn
    presResul = results['mean_test_score']
    weight_index = 0

    plt.xlabel('Neighbors')
    plt.ylabel('Cross-Validated Precisio')
    plt.title('GridSearchCV')

    for weight, color in zip(weight_options, ['g', 'k']):
        plt.plot(k, presResul[weight_index: k.size * len(weight_options): len(weight_options)], color=color, label="%s %s" % ('Weight', weight))
        weight_index += 1

    plt.legend(loc="best")
    plt.show()


# EX2

def mlp_classifier(x_train, y_train):
    param_grid = dict(max_iter=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100], hidden_layer_sizes=(100, 100,),
                      activation=['relu'], solver=['sgd'], learning_rate=['constant'], learning_rate_init=[0.02])
    results = grid_search(MLPClassifier(), param_grid, x_train, y_train)
    means = results['mean_test_score']

    plt.xlabel('Iteracions')
    plt.ylabel('Cross-Validated Precisio')
    plt.title('GridSearchCV')
    plt.plot(param_grid['max_iter'], means[0:len(means):2], 'g')
    plt.show()


def ada_boost(x_train, y_train):
    param_grid = dict(n_estimators=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    results = grid_search(AdaBoostClassifier(), param_grid, x_train, y_train)
    means = results['mean_test_score']

    plt.xlabel('Estimators')
    plt.ylabel('Cross-Validated Precisio')
    plt.title('GridSearchCV')
    plt.plot(param_grid['n_estimators'], means, 'g')
    plt.show()


# EX3

def twenty():
    twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
    twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    text_clf.fit(twenty_train.data, twenty_train.target)

    docs_test = twenty_test.data
    predicted = text_clf.predict(docs_test)
    mean = np.mean(predicted == twenty_test.target)
    print "Mean: %.2f%%" % mean


def grid_search(classifier, parameters, x_train, y_train):
    clf = GridSearchCV(classifier, parameters, cv=10, scoring='accuracy', iid=False, n_jobs=-1)
    clf.fit(x_train, y_train)
    return clf.cv_results_




# Script

# Descarreguem dataset
digits = sklearn.datasets.load_digits()
x = digits.data
y = digits.target
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.50, random_state=0)

# GridSearch amb KNeighborsClassifier
#knn(x_train, y_train)

# GridSearch amb MLPClassifier
#mlp_classifier(x_train, y_train)

# GridSearch amb AdaBoost
#ada_boost(x_train, y_train)

# 20newsgroup amb bayes
twenty()
