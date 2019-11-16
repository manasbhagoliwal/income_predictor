import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
dataclean = np.genfromtxt("../data/adult.data.cleaned2", dtype='str')  # created numpy with string data
a = np.zeros((1001,15))
A = a.astype(str)
count = 0
# in this loop removing all nonUS born from the array
#for i in range(len(dataclean)):
#    if dataclean[i][13] == "United-States,":
##        A[count]=dataclean[i]
 #       count = count+1

for i in range(len(dataclean)):
        A[count]=dataclean[i]
        count = count+1
#print(A[0:,0:])

df = pd.DataFrame(data=A[1:,0:],columns=A[0,0:])
#print(df)
#my_dict = {"Private,":"1","Self-emp-not-inc,":"2","Self-emp-inc,":"3", "Federal-gov,":"4","Local-gov,":"5","State-gov,":"6","Without-pay,":"7","Never-worked,":"8"}
labelsEdu = df['education'].astype('category').cat.categories.tolist()
labelsWork = df['workclass'].astype('category').cat.categories.tolist()
labelsMarry = df['marital-status'].astype('category').cat.categories.tolist()
labelsOccup = df['occupation'].astype('category').cat.categories.tolist()
labelsFamily = df['relationship'].astype('category').cat.categories.tolist()
labelsRace = df['race'].astype('category').cat.categories.tolist()
labelsSex = df['sex'].astype('category').cat.categories.tolist()
labelsCountry = df['native-country'].astype('category').cat.categories.tolist()
labelsIncome = df['income'].astype('category').cat.categories.tolist()
replace_map_comp = {'education' : {k: v for k,v in zip(labelsEdu,list(range(1,len(labelsEdu)+1)))}, 'workclass' : {k: v for k,v in zip(labelsWork,list(range(1,len(labelsWork)+1)))}, 'marital-status' : {k: v for k,v in zip(labelsMarry,list(range(1,len(labelsMarry)+1)))},
                    'occupation' : {k: v for k,v in zip(labelsOccup,list(range(1,len(labelsOccup)+1)))}, 'relationship' : {k: v for k,v in zip(labelsFamily,list(range(1,len(labelsFamily)+1)))}, 'race' : {k: v for k,v in zip(labelsRace,list(range(1,len(labelsRace)+1)))},
                    'sex' : {k: v for k,v in zip(labelsSex,list(range(1,len(labelsSex)+1)))}, 'native-country' : {k: v for k,v in zip(labelsCountry,list(range(1,len(labelsCountry)+1)))},'income' : {k: v for k,v in zip(labelsIncome,list(range(1,len(labelsIncome)+1)))}}

df_replace = df.copy()
df_replace=df_replace.replace(replace_map_comp)
#getting rid of commas in initially numerical values
df_replace['age'] = df_replace['age'].str.replace(',', '')
df_replace['fnlwgt'] = df_replace['fnlwgt'].str.replace(',', '')
df_replace['education-num'] = df_replace['education-num'].str.replace(',', '')
df_replace['capital-gain'] = df_replace['capital-gain'].str.replace(',', '')
df_replace['capital-loss'] = df_replace['capital-loss'].str.replace(',', '')
df_replace['hours-per-week'] = df_replace['hours-per-week'].str.replace(',', '')
df_replace.astype('int64').dtypes
print(df_replace)
# Split the data into training and test sets
#X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(digits.data, digits.target, digits.images, test_size=0.25, random_state=42)

#print(df_replace.info())  good for debug!!!
#print(df_replace.dtypes) good for debug!!!
