# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu).
#


import random
import numpy as np
import matplotlib.pyplot as plt

# A simple utility class to load data sets for this assignment
class DataSet:
    def __init__(self, data_set):
        """
        Initialize a data set and load both training and test data
        DO NOT MODIFY THIS FUNCTION
        """
        self.name = data_set

        # The training and test labels
        self.labels = {'train': None, 'test': None}

        # The training and test examples
        self.examples = {'train': None, 'test': None}

        # Load all the data for this data set
        for data in ['train', 'test']:
            self.load_file(data)

        # The shape of the training and test data matrices
        self.num_train = self.examples['train'].shape[0]
        self.num_test = self.examples['test'].shape[0]
        self.dim = self.examples['train'].shape[1]

    def load_file(self, dset_type):
        """
        Load a training set of the specified type (train/test). Returns None if either the training or test files were
        not found. NOTE: This is hard-coded to use only the first seven columns, and will not work with all data sets.
        DO NOT MODIFY THIS FUNCTION
        """
        path = './data/{0}.{1}'.format(self.name, dset_type)
        try:
            file_contents = np.genfromtxt(path, missing_values=0, skip_header=0, delimiter=',',
                                          usecols=range(0,22), dtype=int)

            self.labels[dset_type] = file_contents[:, 0]
            self.examples[dset_type] = file_contents[:, 1:]

        except RuntimeError:
            print('ERROR: Unable to load file ''{0}''. Check path and try again.'.format(path))


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """
    temp = {
    }
    arr, count = np.unique(x, return_counts=True)
    for i in arr:
        temp[i] = []
        for j in range(len(x)):
            if i == x[j]:
                temp[i].append(j)
    return temp



def entropy(y, weight):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    h = 0
    temp = {
        0: 0,
        1: 0
        }
    y_len = len(y)
    if y_len != 0:
        for i in range(y_len):
            if y[i] == 0:
                temp[0] = temp[0] + weight[i]
            elif y[i] == 1:
                temp[1] = temp[1] + weight[i]
        sum = temp[0] + temp[1]
        for j in range(len(temp)):
            temp[j] = temp[j]/sum
            if temp[j] != 0:
                h = temp[j] * np.log2(temp[j]) + h
        return -h
    else:
        return 0

def mutual_information(x, y, weight):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    h_y = entropy(y, weight)
    x_partition = partition(x)
    temp = 0
    total_weight = 0
    for j in x_partition:
        weight_i = np.sum(weight[x_partition[j]])
        temp = ((weight_i) * entropy(y[x_partition[j]], weight[x_partition[j]])) + temp
        total_weight = weight_i + total_weight
    h_y_of_x = temp / total_weight
    return (h_y - h_y_of_x)

def id3(x, y, attributes, max_depth, weight, depth=0):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of attributes
    to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attributes is empty (there is nothing to split on), then return the most common value of y
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y
    Otherwise the algorithm selects the next best attribute using INFORMATION GAIN as the splitting criterion and
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    See https://gkunapuli.github.io/files/cs6375/04-DecisionTrees.pdf (Slide 18) for more details.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current level.
    * The subtree itself can be nested dictionary, or a single label (leaf node).

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1): 1,
     (4, 2): {(3, 1): 0,
              (3, 2): 0,
              (3, 3): 0},
     (4, 3): {(5, 1): 0,
              (5, 2): 0},
     (4, 4): {(0, 1): 0,
              (0, 2): 0,
              (0, 3): 1}}
    """


    tree = {}
    arr, count = np.unique(y, return_counts=True)
    ''' return the max of label if attribute array is empty or depth of the current itireation
        reaches the specified max depth of the tree or when input array is empty'''
    if len(attributes) == 0 or depth == max_depth or len(x) == 0:
        return arr[np.argmax(count)]
    elif len(arr) == 1:
        ''' return 1 when all the values of label list are one'''
        return arr[0]
    else:
        ''' if none of the above cases matches then find the best attribute to split and call id3 recursively'''
        informationGain = get_mutual_information(x, y, attributes, weight)
        bestAttr, bestValue = choose_attribute(informationGain)
        a = partition(x[:,bestAttr])
        new_attributes = list(filter(lambda x: x!= (bestAttr, bestValue), attributes))
        non_best_indicies = []
        for i in a:
            if i != bestValue:
                non_best_indicies.extend(a[i])
        depth+=1
        for i in range(0,2):
            if i == 0:
                index = a[bestValue]
                new_x = x[index]
                new_y = y[index]
                tree[bestAttr, bestValue, 'true'] = id3(new_x, new_y, new_attributes, max_depth,weight, depth)
            else:
                new_x = x[non_best_indicies]
                new_y = y[non_best_indicies]
                tree[bestAttr, bestValue, 'false'] = id3(new_x, new_y, new_attributes, max_depth,weight, depth)
    return tree


"""
    choose_attribute is to choose the best attribute which has maximum gain
"""
def choose_attribute(infoGain):
    maxGain = 0
    bestAttrVlaue = 0
    keys = list(infoGain.keys())
    for key in keys:
        gain = infoGain[key]
        if(gain >= maxGain):
            maxGain = gain
            bestAttrVlaue = key
    print('maxgain',maxGain)
    return bestAttrVlaue

def get_mutual_information(x, y, attributes, weight):
    infoGain = {}
    row , col = np.shape(x)

    for attr in range(0, col):
        x_partition = partition(x[:, attr])
        array = x_partition.keys();
        for attribute in attributes:
            temp = []
            key , value = attribute
            if(attr == key) and (value in array):
                indexes = x_partition[value]
                for i in range(0, row):
                    if i in indexes:
                        temp.append(1)
                    else:
                        temp.append(0)
                infoGain[(attr, value)] = mutual_information(temp, y, weight)
                if(infoGain[(attr, value)] < 0):
                    print('negative info', infoGain[(attr, value)])
    return infoGain


def predict_label(x, tree):
    keys = list(tree.keys())
    for key in keys:
        attr, value, bool = key
        for i in range(0, len(x)):
            if i == attr:
                if x[i] == value:
                    if type(tree[key]) is dict:
                        return predict_label(x, tree[key])
                    else:
                        return tree[key]
                else:
                    newKey = (attr, value, 'false')
                    if type(tree[newKey]) is dict:
                        return predict_label(x, tree[newKey])
                    else:
                        return tree[newKey]


def predict_example(x, h_ens):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """
    y_preict = []

    for numHypo in h_ens:
        alpha, tree = h_ens[numHypo]
        y = predict_label(x, tree)
        y_preict.append(y)
    arr, count = np.unique(y_preict, return_counts=True)
    return arr[np.argmax(count)]

def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    count = 0;
    label_len = len(y_true)
    for i in range(0, label_len):
        if y_pred[i] != y_true[i]:
            count+=1
    return count / label_len



def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def confusion_matrix(trueLabels, predLabels):

    matrix = np.zeros((2,2))
    for i in range(len(trueLabels)):
        if trueLabels[i] == 1 and predLabels[i] == 1:
            matrix[0][0] += 1
        elif trueLabels[i] == 1 and predLabels[i] == 0:
            matrix[0][1] += 1
        elif trueLabels[i] == 0 and predLabels[i] == 1:
            matrix[1][0] += 1
        elif trueLabels[i] == 0 and predLabels[i] == 0:
            matrix[1][1] += 1

    return matrix

def randomFunction(x):
    indexList= []
    length = len(x[:, 1])
    for i in range(0, length):
        index = random.randint(0, length-1)
        indexList.append(index)
    return indexList

def bagging(x, y, maxdepth, numtrees):
    h_i = {}
    attributes =[]
    rows, cols = np.shape(x)
    for i in range(cols):
        arr = np.unique(x_train[:, i])
        for value in arr:
            attributes.append((i, value))

    weight = np.ones((rows, 1), dtype=int)
    alpha_i = 1
    for i in range(0, numtrees):
        radIndexes = randomFunction(x)
        tree = id3(x[radIndexes], y[radIndexes], attributes, maxdepth, weight)
        h_i[i] = (alpha_i, tree)
    return h_i

def predict_boosting_example(x, h_ens):
    """
    For prediciting exampls with boosting alogirthm where we multiply the precition with respecte to the alpha and normalize it with
    total alpha vlaues

    Returns the predicted label of x according to tree
    """
    y_preict = []
    total_alpha = 0

    for numHypo in h_ens:
        alpha, tree = h_ens[numHypo]
        y = predict_label(x, tree)
        y_preict.append(y*alpha)
        total_alpha += alpha
    predictValue = np.sum(y_preict) / total_alpha

    if(predictValue >= 0.5):
        return 1
    else:
        return 0

def boosting(x, y, max_depth,num_stumps):
    rows, cols = np.shape(x)
    weight = []
    h_ens = {}
    alpha_i = 0
    trn_pred = []
    attributes = []
    for i in range(cols):
        arr = np.unique(x_train[:, i])
        for value in arr:
            attributes.append((i, value))

    for stump in range(0, num_stumps):
        if stump == 0:
            d = (1/rows)
            weight = np.full((rows, 1), d)
        else:
            pre_weight = weight
            weight = []
            for i in range(rows):
                if y[i] == trn_pred[i]:
                    weight.append(pre_weight[i] * np.exp(-1*alpha_i))
                else:
                    weight.append(pre_weight[i] * np.exp(alpha_i))
            d_total = np.sum(weight)
            weight = weight / d_total
        tree = id3(x, y, attributes, max_depth, weight)

        trn_pred = [predict_label(x[i, :], tree) for i in range(rows)]
        temp = 0
        for i in range(rows):
            if(trn_pred[i] != y[i]):
                temp += weight[i]
        err = (1/(np.sum(weight))) * temp
        alpha_i = 0.5 * np.log((1-err)/err)
        h_ens[stump] = (alpha_i, tree)
    return h_ens


if __name__ == '__main__':
    #
    # Below is an example of how a decision tree can be trained and tested on a data set in the folder './data/'. Modify
    # this function appropriately to answer various questions from the Programming Assignment.
    #

    # Load a data set
    data = DataSet('mushroom')

    # Get a list of all the attribute indices
    attribute_idx = np.array(range(data.dim))


    attr_lits = []
    x_train = data.examples['train']
    y_train =  data.labels['train']
    # # making a lits
    # for i in attribute_idx:
    #     arr = np.unique(x_train[:, i])
    #     for value in arr:
    #         attr_lits.append((i, value))

    # # Bagging
    # max_depth = 5
    # num_trees = 20
    # h_ens = bagging(x_train, y_train, max_depth, num_trees)
    #
    #
    #
    #
    #
    # print('Bagging with maxdepth '+ str(max_depth) + ' bagsize ' + str(num_trees))
    #
    # # # Compute the training error for bagging
    # trn_pred = [predict_example(data.examples['train'][i, :], h_ens) for i in range(data.num_train)]
    # trn_err = compute_error(data.labels['train'], trn_pred)
    # print('train_error', trn_err)
    #
    # # Compute the test error for bagging
    # tst_pred = [predict_example(data.examples['test'][i, :], h_ens) for i in range(data.num_test)]
    # tst_err = compute_error(data.labels['test'], tst_pred)
    # print('test_error', tst_err)

    # #######   Boosting ################
    max_depth = 1
    num_stumps = 10

    h_ens = boosting(x_train, y_train, max_depth, num_stumps)

    print('Boosting with maxdepth' + str(max_depth) + 'bagsize' + str(num_stumps))

    # # Compute the training error for bagging
    trn_pred = [predict_boosting_example(data.examples['train'][i, :], h_ens) for i in range(data.num_train)]
    trn_err = compute_error(data.labels['train'], trn_pred)
    print('train_error', trn_err)

    # Compute the test error for bagging
    tst_pred = [predict_boosting_example(data.examples['test'][i, :], h_ens) for i in range(data.num_test)]
    tst_err = compute_error(data.labels['test'], tst_pred)
    print('test_error', tst_err)

    matirx = confusion_matrix(data.labels['test'], tst_pred)

    print("Confunsion matrix For depth "+ str(max_depth) + " bagsize " + str(num_stumps) + "\n" + str(matirx))













