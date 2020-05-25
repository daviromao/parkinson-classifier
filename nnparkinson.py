from database import readDataBaseImage, createDataBase
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import metrics

import scikitplot as skplt
import matplotlib.pyplot as plt

import numpy as np
import pickle

classifier = None

def createAndTrainNN():
    global classifier

    X, y = readDataBaseImage()
    nSamples = len(X)

    X = list(map(lambda image: image.flatten(), X))
    y = list(map(lambda n: n, y))

    classifier = MLPClassifier(activation='relu', alpha=0.001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=[80], learning_rate='constant',
              learning_rate_init=0.001, max_fun=15000, max_iter=200,
              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
              power_t=0.5, random_state=1, shuffle=True, solver='adam',
              tol=0.0001, validation_fraction=0.1, verbose=False,
              warm_start=False)

    classifier.fit(X[: nSamples//2],
                   y[: nSamples//2])

    with open('./_classifiersave.pd', 'wb') as file:
        pickle.dump(classifier, file)

def loadClassifier():
    global classifier
    with open('./_classifiersave.pd', 'rb') as file:
        classifier = pickle.load(file)

def predict(imageMatrix):
    global classifier
    if classifier == None: loadClassifier()

    imageMatrix = imageMatrix.reshape(1, -1)
    predict = classifier.predict(imageMatrix)[0]

    return predict

def testNN():
    global classifier

    X, y = readDataBaseImage()
    nSamples = len(X)
    print(X.shape)
    X = np.array(list(map(lambda image: image.flatten(), X)))
    y = np.array(list(map(lambda n: n, y)))


    expected = y[nSamples//2 :]
    predicts = classifier.predict(X[nSamples//2 :])

    print("Relatório de Classificação :\n", classifier, "\n")
    print(metrics.classification_report(expected, predicts,target_names=['Healthy', 'Patient']))

    print("Matrizes de Confusão: \n{}".format(metrics.confusion_matrix(expected, predicts)))

    y_true = y[ :nSamples//2]
    y_probas = classifier.predict_proba(X[ :nSamples//2])
    skplt.metrics.plot_roc(y_true, y_probas)
    plt.show()


if __name__ == "__main__":
    #createDataBase()
    #createAndTrainNN()
    loadClassifier()
    testNN()
