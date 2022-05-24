from __future__ import annotations

from handler import DataHandler
from decision_tree import ID3

import sys

if __name__ == "__main__":
    #obtain arguments
    train_file_name = sys.argv[1] #path to file containing train data
    test_file_name = sys.argv[2] #path to file containing test data
    depth = None
    if len(sys.argv) > 3:
        depth = int(sys.argv[3])
    
    #prepare train and test data
    handler = DataHandler()
    train_dataset = handler.preprocces_data(train_file_name)
    test_dataset = handler.preprocces_data(test_file_name)
    
    model = ID3(depth)
    model.fit(train_dataset) #fit the model with provided train data
    model.predict(test_dataset) #make predictions on test data