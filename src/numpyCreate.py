import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import copy
import matplotlib.pyplot as plt

#from sklearn.model_selection import KFold
dataclean = np.genfromtxt("../data/adult.data.cleaned2", dtype='str')
a = np.zeros((1001,15))     # we need 'a' to create string numpy 'A' out of it
A = a.astype(str)
count = 0
for i in range(len(dataclean)):
    A[count]=dataclean[i]
    count = count+1
#print(A[0:,0:])

df = pd.DataFrame(data=A[1:,0:],columns=A[0, 0:]) # creating dataframes and setting up the columns

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
replace_map_comp = {'education' : {k: v for k,v in zip(labelsEdu,list(range(1,len(labelsEdu)+1)))}, 'workclass' : {k: v for k,v in zip(labelsWork,list(range(1,len(labelsWork)+1)))}, 'marital-status' : {k: v for k,v in zip(labelsMarry,list(range(1,len(labelsMarry)+1)))},
                    'occupation' : {k: v for k,v in zip(labelsOccup,list(range(1,len(labelsOccup)+1)))}, 'relationship' : {k: v for k,v in zip(labelsFamily,list(range(1,len(labelsFamily)+1)))}, 'race' : {k: v for k,v in zip(labelsRace,list(range(1,len(labelsRace)+1)))},
                    'sex' : {k: v for k,v in zip(labelsSex,list(range(1,len(labelsSex)+1)))}, 'native-country' : {k: v for k,v in zip(labelsCountry,list(range(1,len(labelsCountry)+1)))},'income' : {k: v for k,v in zip(labelsIncome,list(range(1,len(labelsIncome)+1)))}}

# making a copy of our dataframe to work on (it is a good practice)
df_replace = df.copy()
df_replace=df_replace.replace(replace_map_comp)


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

# Split the data(80/20) into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# Split the 70% into two train and validation
Xs_train, Xs_val, ys_train, ys_val = train_test_split(X_train, y_train, test_size=0.30)
Xs_train2, Xs_val2, ys_train2, ys_val2 = train_test_split(Xs_train[:100], ys_train[:100], test_size=0.30)
Xs_train3, Xs_val3, ys_train3, ys_val3 = train_test_split(Xs_train[:50], ys_train2[:50], test_size=0.30)
Xs_train4, Xs_val4, ys_train4, ys_val4 = train_test_split(Xs_train[:10], ys_train[:10], test_size=0.30)


print(len(X_train))
print(len(Xs_train))
print(len(Xs_train2))
print(len(Xs_train3))

# C_list = [0.1, 1, 2, 3, 4, 5, 10]
C_list = [0.1, 1, 2, 10]
best_err = 1.1  # Any value greater than 1
best_err2 = 1.1
best_err3 = 1.1
best_err4 = 1.1
best_C = 0.0
best_C2 = 0.0
best_C3 = 0.0
best_C4 = 0.0

for C in C_list:
    svclassifier = SVC(kernel='linear', C=C)
    svclassifier2 = SVC(kernel='linear', C=C)
    svclassifier3 = SVC(kernel='linear', C=C)
    svclassifier4 = SVC(kernel='linear', C=C)

    # Sample 1
    svclassifier.fit(Xs_train, ys_train)
    y_pred = svclassifier.predict(Xs_val)
    err = np.mean(ys_val != np.array([y_pred]).T)
    # Sample 2
    svclassifier2.fit(Xs_train2, ys_train2)
    y_pred2 = svclassifier2.predict(Xs_val2)
    err2 = np.mean(ys_val2 != np.array([y_pred2]).T)
    # Sample 3
    svclassifier3.fit(Xs_train3, ys_train3)
    y_pred3 = svclassifier3.predict(Xs_val3)
    err3 = np.mean(ys_val3 != np.array([y_pred3]).T)
    # Sample 4
    svclassifier4.fit(Xs_train4, ys_train4)
    y_pred4 = svclassifier4.predict(Xs_val4)
    err4 = np.mean(ys_val4 != np.array([y_pred4]).T)

    print("C=", C, ", err=", err)
    print("C=", C, ", err2=", err2)
    print("C=", C, ", err3=", err3)
    print("C=", C, ", err4=", err4)

    if err < best_err:
        best_err = err
        best_C = C
    if err2 < best_err2:
        best_err2 = err2
        best_C2 = C
    if err3 < best_err3:
        best_err3 = err3
        best_C3 = C
    if err4 < best_err4:
        best_err4 = err4
        best_C4 = C

print("best_C=", best_C)
print("best_C=", best_C2)
print("best_C=", best_C3)
print("best_C=", best_C4)

svclassifier = SVC(kernel='linear', C=best_C)
svclassifier2 = SVC(kernel='linear', C=best_C2)
svclassifier3 = SVC(kernel='linear', C=best_C3)
svclassifier4 = SVC(kernel='linear', C=best_C4)

svclassifier.fit(X_train, y_train)
svclassifier2.fit(Xs_train, ys_train)
svclassifier3.fit(Xs_train2, ys_train2)
svclassifier4.fit(Xs_train3, ys_train3)
y_score = svclassifier.score(X_test, y_test)
y_score2 = svclassifier2.score(X_test, y_test)
y_score3 = svclassifier3.score(X_test, y_test)
y_score4 = svclassifier4.score(X_test, y_test)

# Print sample size vs accuracy graph
sizes = [len(X_train), len(Xs_train), len(Xs_train2), len(Xs_train3)]  # x-values
accuracies = [y_score, y_score2, y_score3, y_score4]  # y-values
plt.plot(sizes, accuracies)
plt.xlabel("Sample Size")
plt.ylabel("Accuracy")
plt.title("Sample Size x Accuracy Plot")
plt.show()


bestacc=0
bestacc2=0
bestacc3=0
bestacc4=0

bestSvclassifier = 0
bestSvclassifier2 = 0
bestSvclassifier3 = 0
bestSvclassifier4 = 0


#10-k fold cross-validation

k = 3
for i in range(0, k):
    T = set(range(int(np.floor((800 * i) / k)), int(np.floor(((800 * (i + 1)) / k) - 1)) + 1))
    T2 = set(range(int(np.floor((560 * i) / k)), int(np.floor(((560 * (i + 1)) / k) - 1)) + 1))
    T3 = set(range(int(np.floor((70 * i) / k)), int(np.floor(((70 * (i + 1)) / k) - 1)) + 1))
    T4 = set(range(int(np.floor((35 * i) / k)), int(np.floor(((35 * (i + 1)) / k) - 1)) + 1))
    S = set(range(0, 800)) - T
    S2 = set(range(0, 560)) - T2
    S3 = set(range(0, 70)) - T3
    S4 = set(range(0, 35)) - T4

    # Sample 1
    svclassifier.fit(X_train[list(T)], y_train[list(T)])    # SVM learning every partition
    ypred = svclassifier.predict(X_train[list(S)])
    # accuracy = accuracy_score(y_train[list(S)], ypred)
    accuracy = svclassifier.score(X_train[list(S)], y_train[list(S)])
    # Sample 2
    svclassifier2.fit(Xs_train[list(S2)], ys_train[list(S2)])    # SVM learning every partition
    ypred2 = svclassifier2.predict(Xs_train[list(S2)])
    # accuracy2 = accuracy_score(ys_train[list(S2)], ypred2)
    accuracy2 = svclassifier2.score(Xs_train[list(S2)], ys_train[list(S2)])
    # Sample 3
    svclassifier3.fit(Xs_train2[list(S3)], ys_train2[list(S3)])    # SVM learning every partition
    ypred3 = svclassifier3.predict(Xs_train2[list(S3)])
    # accuracy3 = accuracy_score(ys_train2[list(S3)], ypred3
    accuracy3 = svclassifier3.score(Xs_train2[list(S3)], ys_train2[list(S3)])
    # Sample 4
    svclassifier4.fit(Xs_train3[list(S4)], ys_train3[list(S4)])    # SVM learning every partition
    ypred4 = svclassifier4.predict(Xs_train3[list(S4)])
    # accuracy4 = accuracy_score(ys_train3[list(S4)], ypred4)
    accuracy4 = svclassifier4.score(Xs_train3[list(S4)], ys_train3[list(S4)])

    if accuracy > bestacc:
        bestacc = accuracy
        bestSvclassifier = copy.copy(svclassifier)
    if accuracy2 > bestacc2:
        bestacc2 = accuracy2
        bestSvclassifier2 = copy.copy(svclassifier2)
    if accuracy3 > bestacc3:
        bestacc3 = accuracy3
        bestSvclassifier3 = copy.copy(svclassifier3)
    if accuracy4 > bestacc4:
        bestacc4 = accuracy4
        bestSvclassifier4 = copy.copy(svclassifier4)


print(bestacc)
print(bestacc2)
print(bestacc3)
print(bestacc4)

# Print sample size vs accuracy graph
sizes = [len(X_train), len(Xs_train), len(Xs_train2), len(Xs_train3)]  # x-values
accuracies = [bestacc, bestacc2, bestacc3, bestacc4]  # y-values
plt.plot(sizes, accuracies)
plt.xlabel("Sample Size")
plt.ylabel("Accuracy")
plt.title("Sample Size x Accuracy Plot")
plt.show()

# svclassifier = SVC(kernel='linear')
# svclassifier.fit(X_train, y_train)
y_pred = bestSvclassifier.predict(X_test)               # test the generated model with the test set
print(confusion_matrix(y_test,y_pred))              # print the analysis report
print(classification_report(y_test,y_pred))


