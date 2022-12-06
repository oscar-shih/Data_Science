import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import export_text
import copy
def handle_str(x, Type):
    if Type == 'Car Type':
        if x == "Family":
            return 0
        elif x == "Sports":
            return 1
        else:
            return 2
    elif Type == 'Shirt Size':
        if x == "Small":
            return 0
        elif x == "Medium":
            return 1
        elif x == "Large":
            return 2
        else:
            return 3
    elif Type == "Gender":
        if x == "M":
            return 1
        else:
            return 0
    else:
        if x == "C0":
            return 0
        else :
            return 1
df = pd.read_csv("data.csv").dropna(axis=0)
df.index._data = np.array(df["Customer ID"])
df = df.drop(axis=1, columns="Customer ID")
unprocess_df = copy.deepcopy(df)
for col in df.columns:
    df.loc[:, col] = df.loc[:, col].apply(lambda x: handle_str(x, Type=col))
# df.loc[:, 'Gender '] = df.loc[:, 'Gender '].apply(lambda x: 1 if x == "M " else 0)
# df.loc[:, 'Gender '] = df.loc[:, 'Gender '].apply(lambda x: 1 if x == "M " else 0)
X = df.drop(axis=1, columns='Class')
Y = df['Class']
clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(X, Y)
tree.plot_tree(clf, feature_names=list(X.columns), class_names=['C0', 'C1'])
r = export_text(clf, feature_names=list(X.columns))
print(r)