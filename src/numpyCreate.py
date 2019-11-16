import pandas as pd
import numpy as np
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
df = pd.DataFrame(data=A[1:,1:],columns=A[0,1:])
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
                    'sex' : {k: v for k,v in zip(labelsSex,list(range(1,len(labelsSex)+1)))}, 'native-country' : {k: v for k,v in zip(labelsCountry,list(range(1,len(labelsCountry)+1)))},'income' : {k: v for k,v in zip(labelsIncome,list(range(1,len(labelsIncome)+1)))} }
print(replace_map_comp)
df_replace = df.copy()
df_replace=df_replace.replace(replace_map_comp)
print(df_replace)

#print(df_replace.info())  good for debug!!!

