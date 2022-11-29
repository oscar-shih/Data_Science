import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

x = pd.read_csv('./hw3_Data1/gene.txt', delimiter = ' ', header = None).T
y = pd.read_csv('./hw3_Data1/label.txt', header = None).to_numpy()
y = (y>0).astype(int).reshape(y.shape[0])
x = (x - x.min()) / (x.max() - x.min())
indexes = pd.read_csv('./hw3_Data1/index.txt', delimiter = '\t', header = None)

bestfeatures = SelectKBest(score_func=f_regression, k=5)
fit = bestfeatures.fit(x, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)


featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs','Score']  
print(featureScores.nlargest(10, 'Score'))  # print 10 best features
features = featureScores.nlargest(10, 'Score')['Specs'].to_list()

print(features)
index = indexes.to_numpy()
# print(index[features])
with open("p1.txt", "w", encoding="utf-8") as f:
    for i in range(len(features)):
        f.write("Index " + str(features[i]) + ' ')
        for j in range(len(index[features[i]])):
            f.write(str(index[features[i]][j]) + ' ')
        f.write("\n")