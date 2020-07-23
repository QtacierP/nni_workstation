import torch
import numpy as np

def classify(predict, thresholds=[0, 1, 2, 3, 4]):
    predict = max(predict, thresholds[0])
    for i in reversed(range(len(thresholds))):
        if predict >= thresholds[i]:
            return i

def accuracy(predictions, targets, c_matrix=None, supervised=False, regression=False):
    predictions = predictions.data
    targets = targets.data
    if not regression:
        # avoid modifying origin predictions
        predicted = torch.tensor(
            [torch.argmax(p) for p in predictions]
        ).cuda().long()
    else:

        predicted =  torch.tensor(
        [classify(p.item()) for p in predictions]
    ).cuda().float()

    # update confusion matrix
    if c_matrix is not None:
        for i, p in enumerate(predicted):
            c_matrix[int(targets[i])][int(p.item())] += 1

    correct = (predicted == targets).sum().item()
    if supervised:
        return correct / len(predicted), correct
    else:
        return correct / len(predicted)

def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight, weight_per_class

def quadratic_weighted_kappa(conf_mat):
    assert conf_mat.shape[0] == conf_mat.shape[1]
    cate_num = conf_mat.shape[0]

    # Quadratic weighted matrix
    weighted_matrix = np.zeros((cate_num, cate_num))
    for i in range(cate_num):
        for j in range(cate_num):
            weighted_matrix[i][j] = 1 - float(((i - j)**2) / ((cate_num - 1)**2))

    # Expected matrix
    ground_truth_count = np.sum(conf_mat, axis=1)
    pred_count = np.sum(conf_mat, axis=0)
    expected_matrix = np.outer(ground_truth_count, pred_count)

    # Normalization
    conf_mat = conf_mat / conf_mat.sum()
    expected_matrix = expected_matrix / expected_matrix.sum()

    observed = (conf_mat * weighted_matrix).sum()
    expected = (expected_matrix * weighted_matrix).sum()
    return (observed - expected) / (1 - expected)

def seg_accuracy(predictions, targets, supervised=False, regression=False):
    if len(predictions.shape) == 4 and predictions.shape[1] > 1:
        predicted = torch.argmax(predictions, dim=1).cuda().long().flatten()
    else:
        predicted = predictions.cuda().long().flatten()
        predicted = torch.where(predicted >= 0.5, torch.ones_like(predicted),
                                torch.zeros_like(predicted))
    targets = targets.data.flatten()
    correct = (predicted == targets).sum().item()
    if supervised:
        return correct / len(predicted), correct
    else:
        return correct / len(predicted)


def dice(y_true: torch.Tensor, y_pred: torch.Tensor, target=0, supervised=False) -> torch.Tensor:
    # Concert to flatten tensor
    if len(y_pred.shape) == 4 and y_pred.shape[1] > 1:
        y_pred = torch.argmax(y_pred, dim=1)  # Get label map
    else:
        y_pred = torch.where(y_pred >= 0.5, torch.ones_like(y_pred),
                             torch.zeros_like(y_pred))
    target_pred = torch.where(y_pred == target,
                              torch.ones_like(y_pred),
                              torch.zeros_like(y_pred)).flatten()
    target_true = torch.where(y_true == target,
                              torch.ones_like(y_true),
                              torch.zeros_like(y_true)).flatten()

    tp = (target_true * target_pred).sum().to(torch.float32)
    tn = ((1 - target_true) * (1 - target_pred)).sum().to(torch.float32)
    fp = ((1 - target_true) * target_pred).sum().to(torch.float32)
    fn = (target_true * (1 - target_pred)).sum().to(torch.float32)

    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    if supervised:
        return tp, \
               tn, \
               fp, \
               fn, f1
    else:
        return f1