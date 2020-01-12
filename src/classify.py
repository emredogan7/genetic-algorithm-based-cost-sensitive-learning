from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


def classify(option, X_train, X_test, y_train, y_test):
    if option == 'knn':
        model = KNeighborsClassifier(5)
    elif option == 'nn':
        model = MLPClassifier(
            alpha=1e-2, hidden_layer_sizes=(100), random_state=1)
    elif option == 'dt':
        model = tree.DecisionTreeClassifier()
    else:
        model = MLPClassifier(
            alpha=1e-2, hidden_layer_sizes=(100), random_state=1)

    model.fit(X_train, y_train)
    y_est_train = model.predict(X_train)
    y_est_test = model.predict(X_test)

    f1minScore = min(f1_score(y_test, y_est_test, average=None))
    accuracy = accuracy_score(y_test, y_est_test)

    #target_names = ['class 1', 'class 2', 'class 3']
    #report = classification_report(y_test, y_est_test, target_names=target_names)

    return f1minScore, accuracy
