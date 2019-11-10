to_remove = ['?']

with open('data/adult.data') as oldData, open('data/adult.data.cleaned', 'w') as newData:
    for line in oldData:
        if not any(word in line for word in to_remove):
            newData.write(line)
