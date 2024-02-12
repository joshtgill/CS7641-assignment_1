import numpy as np
import matplotlib.pyplot as plt
import time
import movies as movies
from sklearn import tree, neural_network, ensemble, neighbors, svm, preprocessing



def get_data(file_name, custom_function=None):
    if custom_function:
        return custom_function()

    data = []

    with open(file_name, 'r') as file:
        for line in file.readlines()[1:]:
            row = line.strip().split(',')
            data.append(row)

    data = np.array(data, dtype=np.float64)
    np.random.shuffle(data)
    return data


def train(data, split, classifier, do_scale=False):
    data_size = len(data)
    split_index = int(split * data_size)

    X = data[: split_index,: -1]
    Y = data[: split_index, -1]
    x = data[-(data_size - split_index) :,: -1]
    y = data[-(data_size - split_index) :, -1]

    t = time.time()
    if do_scale:
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
        scaler = preprocessing.StandardScaler().fit(x)
        x = scaler.transform(x)

    classifier = classifier.fit(X, Y)
    P = classifier.predict(X)
    p = classifier.predict(x)

    return np.sum(P != Y) / len(P), np.sum(p != y) / len(p), time.time() - t


def do_learning(inp):
    np.random.seed(0)

    label = inp.get('label')
    data = get_data(inp.get('file'), inp.get('custom_function', None))


    method = 'Decision Tree'
    print(label, method)
    x = []
    y1 = []
    y2 = []
    ts = []
    for i in [x / 100.0 for x in range(10, 100, 10)]:
        classifier = tree.DecisionTreeClassifier(max_depth=8, min_samples_split=2 if label == 'movies' else 10, random_state=0)
        P, p, t = train(data, i, classifier)
        y1.append(P)
        y2.append(p)
        ts.append(t)
        x.append(i)

    plt.title(f'{label.capitalize()} - training vs test error rates ({method})')
    plt.xlabel('Percent trained')
    plt.ylabel('Error rate')
    plt.plot(x, y1, color='blue', label='Training data')
    plt.plot(x, y2, color='green', label='Test data')
    plt.legend(loc="upper right")
    plt.savefig(f'plots/{label}-decision_tree-error.png')
    plt.clf()
    plt.title(f'{label.capitalize()} - training vs test time ({method})')
    plt.xlabel('Percent trained')
    plt.ylabel('Time')
    plt.plot(x, ts, color='blue', label='Training data')
    plt.savefig(f'plots/{label}-decision_tree-error-time.png')
    plt.clf()

    print(label, method)
    x = []
    y1 = []
    y2 = []
    ts = []
    for i in range(1, 51, 2):
        classifier = tree.DecisionTreeClassifier(max_depth=i, min_samples_split=2 if label == 'movies' else 10, random_state=0)
        P, p, t = train(data, 0.8, classifier)
        y1.append(P)
        y2.append(p)
        ts.append(t)
        x.append(i)

    plt.title(f'{label.capitalize()} - validation curve ({method})')
    plt.xlabel('max_depth')
    plt.ylabel('Error rate')
    plt.plot(x, y1, color='blue', label='Training data')
    plt.plot(x, y2, color='green', label='Test data')
    plt.legend(loc="upper right")
    plt.savefig(f'plots/{label}-decision_tree-validation_curve.png')
    plt.clf()
    plt.title(f'{label.capitalize()} - validation curve time ({method})')
    plt.xlabel('max_depth')
    plt.ylabel('Time')
    plt.plot(x, ts, color='blue', label='Training data')
    plt.savefig(f'plots/{label}-decision_tree-validation_curve-time.png')
    plt.clf()



    method = 'Neural Network'
    print(label, method)
    x = []
    y1 = []
    y2 = []
    ts = []
    for i in [x / 100.0 for x in range(10, 100, 10)]:
        P, p, t = train(data, i, neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3,), random_state=0, max_iter=5000), do_scale=True)
        y1.append(P)
        y2.append(p)
        ts.append(t)
        x.append(i)

    plt.title(f'{label.capitalize()} - training vs test error rates ({method})')
    plt.xlabel('Percent trained')
    plt.ylabel('Error rate')
    plt.plot(x, y1, color='blue', label='Training data')
    plt.plot(x, y2, color='green', label='Test data')
    plt.legend(loc="upper right")
    plt.savefig(f'plots/{label}-neural_net-error.png')
    plt.clf()
    plt.title(f'{label.capitalize()} - training vs test time ({method})')
    plt.xlabel('Percent trained')
    plt.ylabel('Time')
    plt.plot(x, ts, color='blue', label='Training data')
    plt.savefig(f'plots/{label}-neural_net-error-time.png')
    plt.clf()

    print(label, method)
    x = []
    y1 = []
    y2 = []
    ts = []
    for i in range(1, 6):
        P, p, t = train(data, 0.80, neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(i,), random_state=0, max_iter=5000), do_scale=True)
        y1.append(P)
        y2.append(p)
        ts.append(t)
        x.append(i)

    plt.title(f'{label.capitalize()} - validation curve ({method})')
    plt.xlabel('hidden layer (x, x)')
    plt.ylabel('Error rate')
    plt.plot(x, y1, color='blue', label='Training data')
    plt.plot(x, y2, color='green', label='Test data')
    plt.legend(loc="upper right")
    plt.savefig(f'plots/{label}-neural_net-validation_curve.png')
    plt.clf()
    plt.title(f'{label.capitalize()} - validation curve time ({method})')
    plt.xlabel('max_depth')
    plt.ylabel('Time')
    plt.plot(x, ts, color='blue', label='Training data')
    plt.savefig(f'plots/{label}-neural_net-validation_curve-time.png')
    plt.clf()



    method = 'Boosted Decision Trees'
    print(label, method)
    x = []
    y1 = []
    y2 = []
    ts = []
    for i in [x / 100.0 for x in range(10, 100, 10)]:
        P, p, t = train(data, i, ensemble.GradientBoostingClassifier(n_estimators=25, learning_rate=1.0, max_depth=9, random_state=0))
        y1.append(P)
        y2.append(p)
        ts.append(t)
        x.append(i)

    plt.title(f'{label.capitalize()} - training vs test error rates ({method})')
    plt.xlabel('Percent trained')
    plt.ylabel('Error rate')
    plt.plot(x, y1, color='blue', label='Training data')
    plt.plot(x, y2, color='green', label='Test data')
    plt.legend(loc="upper right")
    plt.savefig(f'plots/{label}-boosted-error.png')
    plt.clf()
    plt.title(f'{label.capitalize()} - training vs test time ({method})')
    plt.xlabel('Percent trained')
    plt.ylabel('Time')
    plt.plot(x, ts, color='blue', label='Training data')
    plt.savefig(f'plots/{label}-boosted-error-time.png')
    plt.clf()

    print(label, method)
    x = []
    y1 = []
    y2 = []
    ts = []
    for i in range(1, 200, 10):
        P, p, t = train(data, 0.80, ensemble.GradientBoostingClassifier(n_estimators=i, learning_rate=1.0, max_depth=5, random_state=0))
        y1.append(P)
        y2.append(p)
        ts.append(t)
        x.append(i)

    plt.title(f'{label.capitalize()} - validation curve ({method})')
    plt.xlabel('n_estimators')
    plt.ylabel('Error rate')
    plt.plot(x, y1, color='blue', label='Training data')
    plt.plot(x, y2, color='green', label='Test data')
    plt.legend(loc="upper right")
    plt.savefig(f'plots/{label}-boosted-validation_curve.png')
    plt.clf()
    plt.title(f'{label.capitalize()} - validation curve time ({method})')
    plt.xlabel('max_depth')
    plt.ylabel('Time')
    plt.plot(x, ts, color='blue', label='Training data')
    plt.savefig(f'plots/{label}-boosted-validation_curve-time.png')
    plt.clf()



    method = 'SVM'
    print(label, method)
    x = []
    y1 = []
    y2 = []
    ts = []
    for i in [x / 100.0 for x in range(10, 100, 10)]:
        P, p, t = train(data, i, svm.SVC(random_state=0))
        y1.append(P)
        y2.append(p)
        ts.append(t)
        x.append(i)

    plt.title(f'{label.capitalize()} - training vs test error rates ({method})')
    plt.xlabel('Percent trained')
    plt.ylabel('Error rate')
    plt.plot(x, y1, color='blue', label='Training data')
    plt.plot(x, y2, color='green', label='Test data')
    plt.legend(loc="upper right")
    plt.savefig(f'plots/{label}-svc-error.png')
    plt.clf()
    plt.title(f'{label.capitalize()} - training vs test time ({method})')
    plt.xlabel('Percent trained')
    plt.ylabel('Time')
    plt.plot(x, ts, color='blue', label='Training data')
    plt.savefig(f'plots/{label}-svc-error-time.png')
    plt.clf()

    print(label, method)
    x = []
    y1 = []
    y2 = []
    ts = []
    for kernel in ['poly', 'rbf', 'sigmoid']:
        P, p, t = train(data, 0.80, svm.SVC(kernel=kernel, random_state=0))
        y1.append(P)
        y2.append(p)
        ts.append(t)
        x.append(kernel)

    plt.title(f'{label.capitalize()} - validation curve ({method})')
    plt.xlabel('kernel')
    plt.ylabel('Error rate')
    plt.bar(['poly-X', 'rbf-X', 'sigmoid-X'], y1, color='blue', label='Training data')
    plt.bar(['poly-x', 'rbf-x', 'sigmoid-x'], y2, color='green', label='Test data')
    plt.legend(loc="upper right")
    plt.savefig(f'plots/{label}-svc-validation_curve.png')
    plt.clf()
    plt.title(f'{label.capitalize()} - validation curve time ({method})')
    plt.xlabel('max_depth')
    plt.ylabel('Time')
    plt.bar(x, ts, color='blue', label='Training data')
    plt.savefig(f'plots/{label}-svc-validation_curve-time.png')
    plt.clf()



    method = 'k-Nearest Neighbors'
    print(label, method)
    x = []
    y1 = []
    y2 = []
    ts = []
    for i in [x / 100.0 for x in range(10, 100, 10)]:
        P, p, t = train(data, i, neighbors.KNeighborsClassifier(n_neighbors=4))
        y1.append(P)
        y2.append(p)
        ts.append(t)
        x.append(i)

    plt.title(f'{label.capitalize()} - training vs test error rates ({method})')
    plt.xlabel('Percent trained')
    plt.ylabel('Error rate')
    plt.plot(x, y1, color='blue', label='Training data')
    plt.plot(x, y2, color='green', label='Test data')
    plt.legend(loc="upper right")
    plt.savefig(f'plots/{label}-k_nearest-error.png')
    plt.clf()
    plt.title(f'{label.capitalize()} - training vs test time ({method})')
    plt.xlabel('Percent trained')
    plt.ylabel('Time')
    plt.plot(x, ts, color='blue', label='Training data')
    plt.savefig(f'plots/{label}-k_nearest-error-time.png')
    plt.clf()

    print(label, method)
    x = []
    y1 = []
    y2 = []
    ts = []
    for i in range(1, 16):
        P, p, t = train(data, 0.80, neighbors.KNeighborsClassifier(n_neighbors=i))
        y1.append(P)
        y2.append(p)
        ts.append(t)
        x.append(i)

    plt.title(f'{label.capitalize()} - validation curve ({method})')
    plt.xlabel('n_neighbors')
    plt.ylabel('Error rate')
    plt.plot(x, y1, color='blue', label='Training data')
    plt.plot(x, y2, color='green', label='Test data')
    plt.legend(loc="upper right")
    plt.savefig(f'plots/{label}-k_nearest-validation_curve.png')
    plt.clf()
    plt.title(f'{label.capitalize()} - validation curve time ({method})')
    plt.xlabel('n_neighbors')
    plt.ylabel('Time')
    plt.plot(x, ts, color='blue', label='Training data')
    plt.savefig(f'plots/{label}-k_nearest-validation_curve-time.png')
    plt.clf()


def go():
    for inp in [{'label': 'movies', 'file': '', 'custom_function': movies.extract_data},
                {'label': 'credit', 'file': 'data/credit.csv'}]:
        do_learning(inp)


if __name__ == "__main__":
    go()
