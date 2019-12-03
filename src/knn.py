import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
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

#           First find the best k for knn
k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
          31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]

# Split the data(70/30) into training and test sets
X_big, X_test, y_big, y_test = train_test_split(X, y, test_size=0.30)

# Spilt the 70% into two train and validation
X_train, X_val, y_train, y_val = train_test_split(X_big, y_big, test_size=0.30)
X_train2, X_val2, y_train2, y_val2 = train_test_split(X_train, y_train, test_size=0.30)
X_train3, X_val3, y_train3, y_val3 = train_test_split(X_train2, y_train2, test_size=0.30)
X_train4, X_val4, y_train4, y_val4 = train_test_split(X_train3, y_train3, test_size=0.30)

best_err = 1.1  # Any value greater than 1
best_err2 = 1.1  # Any value greater than 1
best_err3 = 1.1  # Any value greater than 1
best_err4 = 1.1  # Any value greater than 1

best_k = 0.0
best_k2 = 0.0
best_k3 = 0.0
best_k4 = 0.0

for k in k_list:
    alg = KNeighborsClassifier(n_neighbors=k, algorithm='brute')
    alg2 = KNeighborsClassifier(n_neighbors=k, algorithm='brute')
    alg3 = KNeighborsClassifier(n_neighbors=k, algorithm='brute')
    alg4 = KNeighborsClassifier(n_neighbors=k, algorithm='brute')

    # Sample 1
    alg.fit(X_train, y_train)
    y_pred = alg.predict(X_val)
    err = np.mean(y_val != np.array([y_pred]).T)
    # Sample 2
    alg2.fit(X_train2, y_train2)
    y_pred2 = alg2.predict(X_val2)
    err2 = np.mean(y_val2 != np.array([y_pred2]).T)
    # Sample 3
    alg3.fit(X_train3, y_train3)
    y_pred3 = alg3.predict(X_val3)
    err3 = np.mean(y_val3 != np.array([y_pred3]).T)
    # Sample 4
    alg4.fit(X_train4, y_train4)
    y_pred4 = alg4.predict(X_val4)
    err4 = np.mean(y_val4 != np.array([y_pred4]).T)

    print("k=", k, ", err=", err)
    print("k=", k, ", err2=", err2)
    print("k=", k, ", err3=", err3)
    print("k=", k, ", err4=", err4)
    if err < best_err:
        best_err = err
        best_k = k
    if err2 < best_err2:
        best_err2 = err2
        best_k2 = k
    if err3 < best_err3:
        best_err3 = err3
        best_k3 = k
    if err4 < best_err4:
        best_err4 = err4
        best_k4 = k

fpr = dict()
tpr = dict()
roc_auc = dict()

# Compute ROC curve and ROC area for each class
for i in range(0, 2):
    fpr[i], tpr[i], labels = roc_curve(y_val, y_pred, 1+i)
    roc_auc[i] = auc(fpr[i], tpr[i])

print("kkkkkkkk")

print(y_test)

print("kkkkkkkk")

print(y_pred)

print(fpr, " auuuu ", tpr)
print(roc_auc)

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_val.ravel(), y_pred.ravel(), 2)
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

print(roc_auc)

plt.figure()
lw = 2
plt.plot(fpr[1], tpr[1], color='red', lw=lw, label='ROC curve (area = %0.8f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

print("best_k=", best_k)
print("best_k2=", best_k2)
print("best_k3=", best_k3)
print("best_k4=", best_k4)

alg = KNeighborsClassifier(n_neighbors=best_k, algorithm='brute')
alg2 = KNeighborsClassifier(n_neighbors=best_k2, algorithm='brute')
alg3 = KNeighborsClassifier(n_neighbors=best_k3, algorithm='brute')
alg4 = KNeighborsClassifier(n_neighbors=best_k4, algorithm='brute')

alg.fit(X_train, y_train)
probs = alg.predict_proba(X_test)
probs = probs[:, 1]
y_pred = alg.predict(X_test)
err = np.mean(y_test != np.array([y_pred]).T)

alg2.fit(X_train2, y_train2)
y_pred2 = alg2.predict(X_test)
err2 = np.mean(y_test != np.array([y_pred2]).T)
alg3.fit(X_train3, y_train3)
y_pred3 = alg3.predict(X_test)
err3 = np.mean(y_test != np.array([y_pred3]).T)
alg4.fit(X_train4, y_train4)
y_pred4 = alg4.predict(X_test)
err4 = np.mean(y_test != np.array([y_pred4]).T)

print(err)
print(err2)
print(err3)
print(err4)


auc = roc_auc_score(y_test, probs)
print('Model: ROC AUC=%.7f' % auc)
fpr, tpr, _ = roc_curve(y_test, probs, 2)
plt.plot(fpr, tpr, marker='.', label='Model')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


#  K fold cross validation using the best k found above


# Split the data(80/20) into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.20)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X_train2, y_train2, test_size=0.20)
X_train4, X_test4, y_train4, y_test4 = train_test_split(X_train3, y_train3, test_size=0.20)

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=best_k, algorithm='brute')
knn2 = KNeighborsClassifier(n_neighbors=best_k2, algorithm='brute')
knn3 = KNeighborsClassifier(n_neighbors=best_k3, algorithm='brute')
knn4 = KNeighborsClassifier(n_neighbors=best_k4, algorithm='brute')

(n, d) = np.shape(X_train)
(n2, d2) = np.shape(X_train2)
(n3, d3) = np.shape(X_train3)
(n4, d4) = np.shape(X_train4)
k = 10
# a list which carries the score from each kfolds
scores_kfold = list()
scores_kfold2 = list()
scores_kfold3 = list()
scores_kfold4 = list()
best_score = 0.0
best_score2 = 0.0
best_score3 = 0.0
best_score4 = 0.0
bfoldknn = 0.0
bfoldknn2 = 0.0
bfoldknn3 = 0.0
bfoldknn4 = 0.0

# kfold with k = 10
for i in range(0, k):
    T = set(range(int(np.floor((n * i) / k)), int(np.floor(((n * (i + 1)) / k) - 1)) + 1))
    T2 = set(range(int(np.floor((n2 * i) / k)), int(np.floor(((n2 * (i + 1)) / k) - 1)) + 1))
    T3 = set(range(int(np.floor((n3 * i) / k)), int(np.floor(((n3 * (i + 1)) / k) - 1)) + 1))
    T4 = set(range(int(np.floor((n4 * i) / k)), int(np.floor(((n4 * (i + 1)) / k) - 1)) + 1))
    S = set(range(0, n)) - T
    S2 = set(range(0, n2)) - T2
    S3 = set(range(0, n3)) - T3
    S4 = set(range(0, n4)) - T4
    temp_model = knn.fit(X_train[list(S)], y_train[list(S)])
    temp_model2 = knn2.fit(X_train2[list(S2)], y_train2[list(S2)])
    temp_model3 = knn3.fit(X_train3[list(S3)], y_train3[list(S3)])
    temp_model4 = knn4.fit(X_train4[list(S4)], y_train4[list(S4)])
    sc = knn.score(X_train[list(T)], y_train[list(T)])
    sc2 = knn2.score(X_train2[list(T2)], y_train2[list(T2)])
    sc3 = knn3.score(X_train3[list(T3)], y_train3[list(T3)])
    sc4 = knn4.score(X_train4[list(T4)], y_train4[list(T4)])

    if sc > best_score:
        bfoldknn = temp_model
        best_score = sc
    if sc2 > best_score2:
        bfoldknn2 = temp_model2
        best_score2 = sc2
    if sc3 > best_score3:
        bfoldknn3 = temp_model3
        best_score3 = sc3
    if sc4 > best_score4:
        bfoldknn4 = temp_model4
        best_score4 = sc4

    scores_kfold.append(sc)
    scores_kfold2.append(sc2)
    scores_kfold3.append(sc3)
    scores_kfold4.append(sc4)

# Print each cv score (accuracy) and average them
print(scores_kfold)
print(scores_kfold2)
print(scores_kfold3)
print(scores_kfold4)
final_score = bfoldknn.score(X_test, y_test)
final_score2 = bfoldknn2.score(X_test, y_test)
final_score3 = bfoldknn3.score(X_test, y_test)
final_score4 = bfoldknn4.score(X_test, y_test)
print("Final score = ", final_score)
print("Final score2 = ", final_score2)
print("Final score3 = ", final_score3)
print("Final score4 = ", final_score4)
npscores = np.asarray(scores_kfold)
npscores2 = np.asarray(scores_kfold2)
npscores3 = np.asarray(scores_kfold3)
npscores4 = np.asarray(scores_kfold4)

print(npscores)
print(npscores2)
print(npscores3)
print(npscores4)


# printing the mean score which represents how our model will perform on new data
print(np.mean(npscores))
print(np.mean(npscores2))
print(np.mean(npscores3))
print(np.mean(npscores4))

# Print sample size vs accuracy graph
sizes = [len(X_train), len(X_train2), len(X_train3), len(X_train4)]  # x-values
accuracies = [np.mean(npscores), np.mean(npscores2), np.mean(npscores3), np.mean(npscores4)]  # y-values
plt.plot(sizes, accuracies)
plt.xlabel("Sample Size")
plt.ylabel("Accuracy")
plt.title("Sample Size x Accuracy Plot")
plt.show()
