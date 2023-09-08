import numpy as np
import pandas as pd

col_names=['sepal_length','sepal_width','petal_length','petal_width','type']
data=pd.read_csv("iris.csv", skiprows=1, header=None, names=col_names)
print(data.head(10))
class Node():
    def __init__(self,feature_index=None,threshold=None,left=None,right=None,info_gain=None,value=None):
        self.feature_index=feature_index
        self.threshold=threshold
        self.left=left
        self.right=right
        self.info_gain=info_gain

        self.value=value

class DecisionTreeClassifier():
    def __init__(self,min_samples_split=2,max_depth=2):
        self.root=None
        self.min_sample_splits=min_samples_split
        self.max_depth=max_depth

    def build_tree(self,dataset,curr_depth=0):

        X,Y=dataset[:,:-1],dataset[:,-1]
        num_samples,num_features=np.shape(X)

        if num_samples>=self.min_sample_splits and curr_depth<=self.max_depth:
            best_split=self.get_best_split(dataset,num_samples,num_features)
            if best_split['info_gain']>0: # =0stands for pure node
                left_subtree=self.build_tree(best_split['dataset_left'],curr_depth+1)
                right_subtree=self.build_tree(best_split['dataset_right'],curr_depth+1)
                return Node(best_split['feature_index'],best_split['threshold'],left_subtree,right_subtree,best_split['info_gain'])

        leaf_value=self.calculate_leaf_value(Y)
        return Node(value=leaf_value)
    '''
    def get_best_split(self,dataset,num_samples,num_features):
        best_split={}
        max_info_gain=-float("inf")

        for feature_index in range(num_features):
            feature_values=dataset[:,feature_index]
            possible_thresholds=np.unique(feature_values)
            for threshold in possible_thresholds:
                dataset_left,dataset_right=self.split(dataset,feature_index,threshold)
                if len(dataset_left) > 0 and len(dataset_left) > 0:
                    y,left_y,right_y=dataset[:-1],dataset_left[:,-1],dataset_right[:,-1]
                    curr_info_gain= self.information_gain(y,left_y,right_y,'gini')
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"]=feature_index
                        best_split["threshold"]=threshold
                        best_split["dataset_left"]=dataset_left
                        best_split["dataset_right"]=dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain=curr_info_gain

                        best_split["feature_index"] = feature_index

        return best_split 
    '''
    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''

        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")

        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    # update the best split if needed
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        # return best split
        return best_split

    def split(self,dataset,feature_index,threshold):
        dataset_left=np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left,dataset_right

    def information_gain(self,parent,l_child,r_child,mode="entropy"):
        weight_l=len(l_child)/len(parent)

        weight_r=len(r_child)/len(parent)
        if mode=='gini':
            gain=self.gini_index(parent)-(weight_l*self.gini_index(l_child)+weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
        return gain
    def entropy(self,y):
        class_labels=np.unique(y)
        entropy=0
        for cls in class_labels:
            p_cls=len(y[y==cls])/len(y)
            entropy+=-p_cls*np.log2(p_cls)
        return entropy


    def gini_index(self,y):
        class_labels = np.unique(y)
        entropy = 1
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls**2
        return entropy

    def calculate_leaf_value(self,y):
        y=list(y)
        return max(y,key=y.count)

    def print_tree(self,tree=None,indent=" "):
        if not tree:
            tree=self.root
        if tree.value is not None:
            print(tree.value)
        else:
            print("X_" + str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    def fit(self,x,y):
        dataset=np.concatenate((x,y),axis=1)
        self.root=self.build_tree(dataset)

    def predict(self,X):
        preditions=[self.make_prediction(x,self.root) for x in X]
        return preditions

    def make_prediction(self,x,tree):
        if tree.value!=None: return tree.value
        feature_val=x[tree.feature_index]
        if feature_val<= tree.threshold:
            return self.make_prediction(x,tree.left)
        else:
            return self.make_prediction(x, tree.right)

X=data.iloc[:,:-1].values
Y=data.iloc[:,-1].values.reshape(-1,1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)

classifier =DecisionTreeClassifier(min_samples_split=3,max_depth=3)
classifier.fit(X_train,Y_train)
classifier.print_tree()


Y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, Y_pred))