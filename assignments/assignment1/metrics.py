def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''

    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score


    assert (len(prediction) == len(ground_truth))

    countOfElements = len(prediction)

    countOfTruthAnswers = 0
    for elementNum in range(countOfElements):
        if prediction[elementNum] == ground_truth[elementNum]:
            countOfTruthAnswers += 1
    accuracy = countOfTruthAnswers/countOfElements

    TPCount = 0
    TNCount = 0
    FPCount = 0
    FNCount = 0
    for elementNum in range(countOfElements):
        if prediction[elementNum] == True and ground_truth[elementNum] == True:
            TPCount += 1
        elif prediction[elementNum] == False and ground_truth[elementNum] == False:
            TNCount += 1
        elif prediction[elementNum] == True and ground_truth[elementNum] == False:
            FPCount += 1
        elif prediction[elementNum] == False and ground_truth[elementNum] == True:
            FNCount += 1

    precision = TPCount / (TPCount + FPCount)

    recall = TPCount / (TPCount + FNCount)

    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy

    assert (len(prediction) == len(ground_truth))

    countOfElements = len(prediction)

    countOfTruthAnswers = 0
    for elementNum in range(countOfElements):
        if prediction[elementNum] == ground_truth[elementNum]:
            countOfTruthAnswers += 1

    accuracy = countOfTruthAnswers / countOfElements

    return accuracy
