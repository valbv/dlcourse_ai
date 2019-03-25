import numpy as np
import matplotlib.pyplot as plt

from dataset import load_svhn
from knn import KNN
from metrics import binary_classification_metrics, multiclass_accuracy

train_X, train_y, test_X, test_y = load_svhn("./data", max_train=10000, max_test=400)

samples_per_class = 5  # Number of samples per class to visualize
plot_index = 1
for example_index in range(samples_per_class):
    for class_index in range(10):
        plt.subplot(5, 10, plot_index)
        image = train_X[train_y == class_index][example_index]
        plt.imshow(image.astype(np.uint8))
        plt.axis('off')
        plot_index += 1
plt.show()

#First, let's prepare the labels and the source data
#Only select 0s and 9s
binary_train_mask = (train_y == 0) | (train_y == 9)
binary_train_X = train_X[binary_train_mask]
binary_train_y = train_y[binary_train_mask] == 0

binary_test_mask = (test_y == 0) | (test_y == 9)
binary_test_X = test_X[binary_test_mask]
binary_test_y = test_y[binary_test_mask] == 0

# Reshape to 1-dimensional array [num_samples, 32*32*3]
binary_train_X = binary_train_X.reshape(binary_train_X.shape[0], -1)
binary_test_X = binary_test_X.reshape(binary_test_X.shape[0], -1)

# Create the classifier and call fit to train the model
# KNN just remembers all the data
knn_classifier = KNN(k=1)
knn_classifier.fit(binary_train_X, binary_train_y)

# # TODO: implement compute_distances_two_loops in knn.py
dists = knn_classifier.compute_distances_two_loops(binary_test_X)
assert np.isclose(dists[0, 10], np.sum(np.abs(binary_test_X[0] - binary_train_X[10])))

# # TODO: implement compute_distances_one_loop in knn.py
dists = knn_classifier.compute_distances_one_loop(binary_test_X)
assert np.isclose(dists[0, 10], np.sum(np.abs(binary_test_X[0] - binary_train_X[10])))

# TODO: implement compute_distances_no_loops in knn.py
dists = knn_classifier.compute_distances_no_loops(binary_test_X)
assert np.isclose(dists[0, 10], np.sum(np.abs(binary_test_X[0] - binary_train_X[10])))

# TODO: implement predict_labels_binary in knn.py
prediction = knn_classifier.predict(binary_test_X)

# TODO: implement binary_classification_metrics in metrics.py
precision, recall, f1, accuracy = binary_classification_metrics(prediction, binary_test_y)
print("KNN with k = %s" % knn_classifier.k)
print("Accuracy: %4.2f, Precision: %4.2f, Recall: %4.2f, F1: %4.2f" % (accuracy, precision, recall, f1))

# Let's put everything together and run KNN with k=3 and see how we do
knn_classifier_3 = KNN(k=3)
knn_classifier_3.fit(binary_train_X, binary_train_y)
prediction = knn_classifier_3.predict(binary_test_X)

precision, recall, f1, accuracy = binary_classification_metrics(prediction, binary_test_y)
print("KNN with k = %s" % knn_classifier_3.k)
print("Accuracy: %4.2f, Precision: %4.2f, Recall: %4.2f, F1: %4.2f" % (accuracy, precision, recall, f1))

##### Cross-validation
# Find the best k using cross-validation based on F1 score
# TODO: split the training data in 5 folds and store them in train_folds_X/train_folds_y
num_folds = 5
numDataInFold = binary_train_y.shape[0] // num_folds

train_folds_X = binary_train_X[ : numDataInFold * num_folds]
train_folds_y = binary_train_y[ : numDataInFold * num_folds]
train_folds_X = train_folds_X.reshape(num_folds, numDataInFold, -1)
train_folds_y = train_folds_y.reshape(num_folds, numDataInFold, -1)

k_choices = [1, 2, 3, 5, 8, 10, 15, 20, 25, 50]
k_to_f1 = {}  # dict mapping k values to mean F1 scores (int -> float)

for k in k_choices:
    # TODO: perform cross-validation
    # Go through every fold and use it for testing and all other folds for training
    # Perform training and produce F1 score metric on the validation dataset
    # Average F1 from all the folds and write it into k_to_f1

    f1SumForAllFols = 0
    for numFold in range(num_folds):
        conditionList = np.full(num_folds, True)
        conditionList[numFold] = False
        binary_train_X_folds = train_folds_X[conditionList]
        binary_train_y_folds = train_folds_y[conditionList]
        binary_train_X_folds = binary_train_X_folds.reshape(binary_train_X_folds.shape[0] * binary_train_X_folds.shape[1], -1)
        binary_train_y_folds = binary_train_y_folds.reshape(binary_train_y_folds.shape[0] * binary_train_y_folds.shape[1], -1)
        binary_test_X_fold = train_folds_X[numFold]
        binary_test_y_fold = train_folds_y[numFold]

        knn_classifier = KNN(k=k)
        knn_classifier.fit(binary_train_X_folds, binary_train_y_folds)
        prediction = knn_classifier.predict(binary_test_X_fold)
        precision, recall, f1, accuracy = binary_classification_metrics(prediction, binary_test_y_fold)
        f1SumForAllFols += f1

    k_to_f1[k] = f1SumForAllFols / num_folds

for k in sorted(k_to_f1):
    print('k = %d, f1 = %f' % (k, k_to_f1[k]))

# TODO Set the best k to the best value found by cross-validation
best_k = max(k_to_f1, key=k_to_f1.get)

best_knn_classifier = KNN(k=best_k)
best_knn_classifier.fit(binary_train_X, binary_train_y)
prediction = best_knn_classifier.predict(binary_test_X)

precision, recall, f1, accuracy = binary_classification_metrics(prediction, binary_test_y)
print("Best KNN with k = %s" % best_k)
print("Accuracy: %4.2f, Precision: %4.2f, Recall: %4.2f, F1: %4.2f" % (accuracy, precision, recall, f1))

##### Now let's use all 10 classes
train_X = train_X.reshape(train_X.shape[0], -1)
test_X = test_X.reshape(test_X.shape[0], -1)

knn_classifier = KNN(k=1)
knn_classifier.fit(train_X, train_y)

# TODO: Implement predict_labels_multiclass
predict = knn_classifier.predict(test_X)

# TODO: Implement multiclass_accuracy
accuracy = multiclass_accuracy(predict, test_y)
print("Accuracy: %4.2f" % accuracy)

# Find the best k using cross-validation based on accuracy
# TODO: split the training data in 5 folds and store them in train_folds_X/train_folds_y
num_folds = 5
numDataInFold = train_y.shape[0] // num_folds

multiclass_train_folds_X = train_X[ : numDataInFold * num_folds]
multiclass_train_folds_y = train_y[ : numDataInFold * num_folds]
multiclass_train_folds_X = multiclass_train_folds_X.reshape(num_folds, numDataInFold, -1)
multiclass_train_folds_y = multiclass_train_folds_y.reshape(num_folds, numDataInFold)

k_choices = [1, 2, 3, 5, 8, 10, 15, 20, 25, 50]
k_to_accuracy = {}

for k in k_choices:
    # TODO: perform cross-validation
    # Go through every fold and use it for testing and all other folds for validation
    # Perform training and produce accuracy metric on the validation dataset
    # Average accuracy from all the folds and write it into k_to_accuracy
    accuracySumForAllFols = 0

    for numFold in range(num_folds):
        conditionList = np.full(num_folds, True)
        conditionList[numFold] = False
        multiclass_train_X_folds = multiclass_train_folds_X[conditionList]
        multiclass_train_y_folds = multiclass_train_folds_y[conditionList]
        multiclass_train_X_folds = multiclass_train_X_folds.reshape(multiclass_train_X_folds.shape[0] * multiclass_train_X_folds.shape[1], -1)
        multiclass_train_y_folds = multiclass_train_y_folds.reshape(multiclass_train_y_folds.shape[0] * multiclass_train_y_folds.shape[1])
        multiclass_test_X_fold = multiclass_train_folds_X[numFold]
        multiclass_test_y_fold = multiclass_train_folds_y[numFold]

        knn_classifier = KNN(k=k)
        knn_classifier.fit(multiclass_train_X_folds, multiclass_train_y_folds)
        prediction = knn_classifier.predict(multiclass_test_X_fold)
        accuracy = multiclass_accuracy(prediction, multiclass_test_y_fold)
        accuracySumForAllFols += accuracy

    k_to_accuracy[k] = accuracySumForAllFols / num_folds

for k in sorted(k_to_accuracy):
    print('k = %d, accuracy = %f' % (k, k_to_accuracy[k]))

#####Финальный тест - классификация на 10 классов на тестовой выборке (test data)
# TODO Set the best k as a best from computed
best_k = max(k_to_accuracy, key=k_to_accuracy.get)

best_knn_classifier = KNN(k=best_k)
best_knn_classifier.fit(train_X, train_y)
prediction = best_knn_classifier.predict(test_X)

# Accuracy should be around 20%!
accuracy = multiclass_accuracy(prediction, test_y)
print("Best KNN with k = %s" % best_k)
print("Accuracy: %4.2f" % accuracy)
