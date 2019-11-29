import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from random import seed
from random import randrange

dataclean = np.genfromtxt("../data/adult.data.cleaned2", dtype='str')
a = np.zeros((1001, 15))  # we need 'a' to create string numpy 'A' out of it
A = a.astype(str)
count = 0
for i in range(len(dataclean)):
    A[count] = dataclean[i]
    count = count + 1
# print(A[0:,0:])

df = pd.DataFrame(data=A[1:, 0:], columns=A[0, 0:])  # creating dataframes and setting up the columns

# creating categories for each column
labelsEdu = df['education'].astype('category').cat.categories.tolist()
labelsWork = df['workclass'].astype('category').cat.categories.tolist()
labelsMarry = df['marital-status'].astype('category').cat.categories.tolist()
labelsOccup = df['occupation'].astype('category').cat.categories.tolist()
labelsFamily = df['relationship'].astype('category').cat.categories.tolist()
labelsRace = df['race'].astype('category').cat.categories.tolist()
labelsSex = df['sex'].astype('category').cat.categories.tolist()
labelsCountry = df['native-country'].astype('category').cat.categories.tolist()
labelsIncome = df['income'].astype('category').cat.categories.tolist()

# convert categorical data into numerical
replace_map_comp = {'education': {k: v for k, v in zip(labelsEdu, list(range(1, len(labelsEdu) + 1)))},
                    'workclass': {k: v for k, v in zip(labelsWork, list(range(1, len(labelsWork) + 1)))},
                    'marital-status': {k: v for k, v in zip(labelsMarry, list(range(1, len(labelsMarry) + 1)))},
                    'occupation': {k: v for k, v in zip(labelsOccup, list(range(1, len(labelsOccup) + 1)))},
                    'relationship': {k: v for k, v in zip(labelsFamily, list(range(1, len(labelsFamily) + 1)))},
                    'race': {k: v for k, v in zip(labelsRace, list(range(1, len(labelsRace) + 1)))},
                    'sex': {k: v for k, v in zip(labelsSex, list(range(1, len(labelsSex) + 1)))},
                    'native-country': {k: v for k, v in zip(labelsCountry, list(range(1, len(labelsCountry) + 1)))},
                    'income': {k: v for k, v in zip(labelsIncome, list(range(1, len(labelsIncome) + 1)))}}

# making a copy of our dataframe to work on (it is a good practice)
df_replace = df.copy()
df_replace = df_replace.replace(replace_map_comp)

# getting rid of commas in initially numerical values
df_replace['age'] = df_replace['age'].str.replace(',', '')
df_replace['fnlwgt'] = df_replace['fnlwgt'].str.replace(',', '')
df_replace['education-num'] = df_replace['education-num'].str.replace(',', '')
df_replace['capital-gain'] = df_replace['capital-gain'].str.replace(',', '')
df_replace['capital-loss'] = df_replace['capital-loss'].str.replace(',', '')
df_replace['hours-per-week'] = df_replace['hours-per-week'].str.replace(',', '')
df_replace.astype('int64').dtypes
print(df_replace)
# print(df_replace.info())  good for debug!!!
# print(df_replace.dtypes) good for debug!!!

df_replace_transposed = df_replace.values.T
y = df_replace_transposed[14]
X = df_replace.values[:1000, :14]
X = np.asarray(X).astype(int)
y = np.asarray(y).astype(int)





#           KNN Implementation









# Split the data(80/20) into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)

(n, d) = np.shape(X_train)
k = 10
# a list which carries the score from each kfolds
scores_kfold = list()

# kfold with k = 10
for i in range(0, k):
    T = set(range(int(np.floor((n * i)/k)), int(np.floor(((n * (i + 1)) / k) - 1)) + 1))
    S = set(range(0, n)) - T
    knn.fit(X_train[list(S)], y_train[list(S)])
    scores_kfold.append(knn.score(X_train[list(T)], y_train[list(T)]))


# Print each cv score (accuracy) and average them
print(scores_kfold)
npscores = np.asarray(scores_kfold)
# printing the mean score which represents how our model will perform on a new unknown data
print(np.mean(npscores))