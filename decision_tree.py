from __future__ import annotations

from handler import DataHandler

import math

class ID3:
    #ID3 class - class for: 1) building a decision tree implemented with Iterative Dichotomiser 3 algorithm, 2) classifying on new data with the built tree
    
    class Node:
        #Node class - inner class used for instancing tree nodes, each node has 2 attributes: feature and a list of subtrees
        def __init__(self, feature: str, subtrees: list):
            self.feature = feature
            self.subtrees = subtrees #a single subtree is a tuple containing a node, and a feature value which leads to the node
    
    #ID3 constructor
    def __init__(self, tree_depth: int = None):
        self._root = None
        self._tree_depth = tree_depth
        self._dataHandler = DataHandler()

    #public method: fit - starts the tree building procedure, stores initial train data in _dataHandler.data, stores the tree in _root attribute and prints the tree's branches 
    #arguments: train_dataset - preproccessed train dataset
    #return values: None
    def fit(self, train_dataset: list) -> None:
        self._dataHandler._data = train_dataset[1:]
        self._root = self._id3(train_dataset[1:], train_dataset[1:], train_dataset[0][:-1], 0)
        self._print_branches(self._get_paths(self._root))

    #private method: _id3 - recursive implementation of Iterative Dichotomiser 3 algorithm for building a decision tree
    #arguments: D - the initial dataset, D_ - dataset for current node, X: list of features, y: list of labels
    #return value: Node created in the current recursion level
    def _id3(self, D: list, D_parent: list, X: list, current_level: int) -> Node:
        labels = dict()
        if len(D) == 0:
            for entrypoint in D_parent:
                if entrypoint[-1] in labels.keys():
                    labels[entrypoint[-1]] = labels.get(entrypoint[-1]) + 1
                else:
                    labels[entrypoint[-1]] = 1
            return self.Node(max(labels, key = labels.get), None)
        
        for entrypoint in D:
            if entrypoint[-1] in labels.keys():
                    labels[entrypoint[-1]] = labels.get(entrypoint[-1]) + 1
            else:
                labels[entrypoint[-1]] = 1
        
        if len(X) == 0:
            return self.Node(max(labels, key = labels.get), None)
        for key in labels.keys():
            if labels.get(key) == len(D):
                return self.Node(key, None)

        if self._tree_depth is not None:
            if current_level == self._tree_depth:
                return self.Node(self._dataHandler.most_frequent_label(D), None)


        information_gains = dict()
        
        for i in range(len(X)):
            values = self._dataHandler.value_extractor(i, D)
            information_gains[X[i]] = self._information_gain(labels, values)
        max_IG_key = max(information_gains, key = information_gains.get)
        max_X = [max_IG_key, information_gains.get(max_IG_key)]

        current_level += 1

        subtrees = []
        values = self._dataHandler.value_extractor(X.index(max_X[0]), D)
        for value in values.keys():
            new_D = []
            for entrypoint in D:
                if entrypoint[X.index(max_X[0])] == value:
                    new_entrypoint = entrypoint.copy()
                    new_entrypoint.pop(X.index(max_X[0]))
                    new_D.append(new_entrypoint)
            X_ = X.copy()
            X_.remove(max_X[0])
            node = self._id3(new_D, D, X_, current_level)
            subtrees.append((value, node))
        return self.Node(max_X[0], subtrees)

    #private method: _get_paths - recursive utility method of fit method for obtaining all paths in a decision tree
    #arguments:  node - current node, branch - name of feature value which leads to the next node, pahts - list containing obtained paths, current_path - list containing the current path
    #return value: list of all paths in a decision tree
    def _get_paths(self, node: Node, branch: str = None, paths: list = None, current_path: list = None) -> list:
        if paths is None:
            paths = []
        if current_path is None:
            current_path = []

        current_path.append(node.feature)
        if node.subtrees is None:
            paths.append(current_path)
        else:
            for child in node.subtrees:
                self._get_paths(child[1], child[0], paths, list(current_path + [child[0]]))
        return paths

    #private method - _print_branches - utility method of fit method for printing all obtained tree paths as branches
    #arguments: paths - list of all obtained tree paths
    #return value: None
    def _print_branches(self, paths: list) -> None:
        paths.sort(key = lambda x: len(x))
        output = "[BRANCHES]:\n"
        for path in paths:
            depth = 1
            for i in range(len(path)):
                if i != len(path) - 1:
                    if i % 2 == 0:
                        if depth == 1:
                            output += str(depth) + ":" + path[i] + "="
                            depth += 1
                        else:
                            output += " " + str(depth) + ":" + path[i] + "="
                            depth += 1
                    else:
                        output += path[i]
                else:
                    output += " " + path[i] + "\n"
        print(output[:-1])

    #private method: _entropy - utility method for the _id3 method, calculates entropy for given dataset
    #arguments: labels and their occurances
    #return value: calculated entropy of a feature value, 0 if only one label is present
    def _entropy(self, values: dict) -> float:
        if len(values.keys()) == 1:
            return 0
        entropy = 0
        for value in values.values():
            entropy += ((-1*value)/(sum(values.values())))*math.log2((value)/(sum(values.values())))
        return entropy

    #private method: _information_gain - utility method for the _id3 method, calculates information gain for given feature
    #arguments: labels and their occurances, values - dictionary of feature values and their label occurances
    #return value: calculated information gain
    def _information_gain(self, total_Y_N: list, values: dict) -> float:
        entropys = 0
        for value in values.values():
            entropys += ((sum(value.values())/sum(total_Y_N.values()))*(self._entropy(value)))
        return self._entropy(total_Y_N) - entropys

    #public method: predict - classifies entrypoints from test data, calculates and prints predictions, accuracy and confusion matrix
    #arguments: test_dataset - preproccessed test dataset
    #return value: None
    def predict(self, test_dataset: list) -> None:
        predictions = []
        correct = 0
        features = test_dataset[0][:-1]
        test_dataset = test_dataset[1:]
        for test in test_dataset:
            node_ = self._root
            test_ = test[:-1]
            for i in range(len(test_)):
                bila_znacajka = False
                if node_.subtrees is not None:
                    for subtree in node_.subtrees:
                        if subtree[0] == test_[features.index(node_.feature)] and node_.feature == features[features.index(node_.feature)]:
                            bila_znacajka = True
                            node_ = subtree[1]
                            if node_.subtrees is None:
                                predictions.append(node_.feature)
                                if node_.feature == test[-1]:
                                    correct += 1
                            break
                    if not bila_znacajka:
                        predictions.append(self._dataHandler.most_frequent_label(None))
                        if correct < 1:
                            correct += 1
                        break
                            
        print("[PREDICTIONS]: " +  " ".join(predictions))
        print("[ACCURACY]: " + format(correct/len(test_dataset), ".5f"))

        labels = list(self._dataHandler.extract_labels(test_dataset))
        labels.sort()
        true_labels = self._dataHandler.extract_label_column(test_dataset)
        confusion_matrix = dict()
        for i in range(len(labels)):
            for j in range(len(labels)):
                confusion_matrix[labels[i] + "," + labels[j]] = 0
        for i in range(len(predictions)):
            confusion_matrix[predictions[i] + "," + true_labels[i]] = confusion_matrix.get(predictions[i] + "," + true_labels[i]) + 1
        print("[CONFUSION_MATRIX]:", end = "\n")
        for i in range(len(labels)):
            for j in range(len(labels)):
                if j != len(labels) - 1:
                    print(str(confusion_matrix.get(labels[j] + "," + labels[i])), end = " ")
                else:
                    print(str(confusion_matrix.get(labels[j] + "," + labels[i])))