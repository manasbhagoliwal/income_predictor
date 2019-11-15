
#import pandas as pd
import numpy as np
#import numpy_indexed as npi
dataclean = np.genfromtxt("../data/adult.data.cleaned", dtype='str')  # created numpy with string data
#df_flights = pd.read_csv("../data/adult.data.cleaned")
a = np.zeros((27504,15))
A = a.astype(str)
count = 0
# in this loop removing all nonUS born from the array
for i in range(len(dataclean)):
    if dataclean[i][13] == "United-States,":
        A[count]=dataclean[i]
        count = count+1

my_dict = {"Federal-gov,":"1","State-gov,":"2","Local-gov,":"3", "Self-emp-inc,":"4","Self-emp-not-inc,":"5","Private,":"6","5th-6th,":"1","United-States,":"hey"}
A = np.vectorize(my_dict.get)(A) # this method is not really good because it erases all non matching values
#remapped_a = npi.remap(dataclean, list(my_dict.keys()), list(my_dict.values()))

print(A)


