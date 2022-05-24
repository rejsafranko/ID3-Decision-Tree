from __future__ import annotations
import csv

class DataHandler:
    #DataHandler class - utility class responsible for data preparation and manipulation"
    
    _data = None
    
    #public method: preprocces_data
    #arguments: file_name - path to file containing data
    #return value: list of entrypoints
    def preprocces_data(self, file_name: str) -> list:
        dataset = []
        file = open(file_name)
        csvreader = csv.reader(file)
        for row in csvreader:
            dataset.append(row)
        file.close()
        return dataset
    
    #public method: extract_labels
    #arguments: dataset - list of entrypoints
    #return value: set of labels in given dataset
    def extract_labels(self, dataset: list) -> set:
        labels = set()
        for entrypoint in dataset:
            labels.add(entrypoint[-1])
        return labels
    
    #public method: extract_label_column
    #arguments: dataset - list of entrypoints
    #return value: label column in given dataset
    def extract_label_column(self, dataset: list) -> list:
        labels = []
        for entrypoint in dataset:
            labels.append(entrypoint[-1])
        return labels
    
    #public method: most_frequent_label
    #arguments: dataset - list of entrypoints
    #return value: label with most occurances in given dataset
    def most_frequent_label(self, data: list) -> str:
        labels = dict()
        if data is None:
            data = self._data
        for entrypoint in data:
            if entrypoint[-1] in labels.keys():
                labels[entrypoint[-1]] = labels.get(entrypoint[-1]) + 1
            else:
                labels[entrypoint[-1]] = 1
        keys = sorted(labels.keys())
        for i in range(len(keys)):
            if i == 0:
                max_key = keys[i]
                max_val = labels.get(keys[i])
            else:
                if labels.get(keys[i]) > max_val:
                    max_key = keys[i]
                    max_val = labels.get(keys[i])
        return max_key

    #public method: value_extractor - extracts single feature's values and counts their occurances
    #arguments: feature_index - index of feature whose values are extracted, dataset - dataset of the current node
    #return value: dictionary of feature values and their occurances
    def value_extractor(self, feature_index: int, dataset: list) -> dict:
        values = dict()
        for entrypoint in dataset:
            values[entrypoint[feature_index]] = dict()
        for entrypoint in dataset:
            if entrypoint[-1] in values.get(entrypoint[feature_index]).keys():
                labels = values.get(entrypoint[feature_index])
                labels[entrypoint[-1]] = labels.get(entrypoint[-1]) + 1
                values[entrypoint[feature_index]] = labels
            else:
                labels = values.get(entrypoint[feature_index])
                labels[entrypoint[-1]] = 1
                values[entrypoint[feature_index]] = labels
        return values