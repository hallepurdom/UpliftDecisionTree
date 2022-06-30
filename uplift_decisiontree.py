# Importing libraries
import numpy as np
import pandas as pd


# Node Class
class Node():
    def __init__(
        self, 
        feature_index=None, 
        threshold=None, 
        left=None, 
        right=None, 
        uplift=None, 
        value=None):
        
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.uplift = uplift

        self.value = value

# Tree Class
class DecisionTreeRegressor():
    def __init__(
        self, 
        max_depth=3,
        min_samples_leaf=1000,
        min_samples_leaf_treated=300,
        min_samples_leaf_control=300
        ):
        
        # initialize the root of the tree 
        self.root = None
        
        # stopping conditions
        self.min_samples_leaf_control = min_samples_leaf_control
        self.max_depth = max_depth
        
    def build_tree(
        self, 
        data,
        current_depth=0):

        X = data[:,:-2]
        Y = data[:,-2]

        num_samples = np.shape(X)[0]
        num_features = np.shape(X)[1]
        best_split = {}
        
        # continue splitting tree until either condition is met
        if num_samples>=self.min_samples_leaf_control and current_depth<=self.max_depth:
            best_split = self.get_best_split(data, num_features)
    
            if best_split["uplift"]>0:
                left_subtree = self.build_tree(best_split["dataset_left"], current_depth+1)
                right_subtree = self.build_tree(best_split["dataset_right"], current_depth+1)
                
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["uplift"])
        
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, data, num_features):
        
        # initialize dictionary
        best_split = {}
        max_uplift = -float("inf")
        # loop over features
        for feature_index in range(num_features):
            
            # threshold algorithm
            column_values = data[:, feature_index]
            unique_values = np.unique(column_values)
            if len(unique_values) >10:
                percentiles = np.percentile(column_values, [3, 5, 10, 20, 30, 50, 70, 80, 90,95, 97])
            else:
                percentiles = np.percentile(unique_values, [10, 50, 90])
            threshold_options = np.unique(percentiles)
            
            # looping through threshold options
            for threshold in threshold_options:
                # get current split
                dataset_left, dataset_right = self.split(data, feature_index, threshold)

                if len(dataset_left)>0 and len(dataset_right)>0:
                    left_y = dataset_left[:, -2]
                    right_y = dataset_right[:, -2]
                    left_tmnt = dataset_left[:, -1]
                    right_tmnt = dataset_right[:, -1]
                    
                    # compute uplift
                    curr_uplift = self.uplift_measure( left_y, right_y, left_tmnt, right_tmnt)
                    # update dictionary
                    if curr_uplift>max_uplift:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["uplift"] = curr_uplift
                        max_uplift = curr_uplift
                        
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def uplift_measure(self, l_child, r_child, left_tmnt, right_tmnt):
        M_left = abs(l_child.mean() - left_tmnt.mean())
        M_right = abs(r_child.mean() - right_tmnt.mean())
        uplift = abs(M_left - M_right)
        return uplift
    
    def calculate_leaf_value(self, Y):
        
        val = np.mean(Y)
        return val
                
    def print_tree(self, tree=None, indent=" "):
        
        if not tree:
            tree = self.root
        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.uplift)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, X, Y, tmnt):
        
        dataset = np.concatenate((X, Y, tmnt), axis=1)
        self.root = self.build_tree(dataset)
        
    def make_prediction(self, x, tree):
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
    
    def predict(self, X):
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions

# Data
example_x = pd.DataFrame(np.load(r'\Users\Halle\Documents\UpliftTree\example_X.npy'))
example_y = pd.DataFrame(np.load(r'C:\Users\Halle\Documents\UpliftTree\example_y.npy'))
example_preds = pd.DataFrame(np.load(r'C:\Users\Halle\Documents\UpliftTree\example_preds.npy'))
example_treatment = pd.DataFrame(np.load(r'C:\Users\Halle\Documents\UpliftTree\example_treatment.npy'))


#Build the tree
tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf = 6000, min_samples_leaf_treated=2500, min_samples_leaf_control=2500)
tree.fit(example_x, example_y, example_treatment)
tree.print_tree()

#Make Predictions
preds = tree.predict(example_x.to_numpy())

df = example_y
df['preds'] = preds
df.head()
df.tail()
df['preds'].unique()