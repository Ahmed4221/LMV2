import pandas as pd

df1 = pd.read_pickle('test_pickle.pkl')

import pandas as pd
import pickle
df2 = pd.read_pickle('training_pickle.pkl')

words = []
labels = []
bbox = []

words.append(df1[0][0])
labels.append(df1[1][0])
bbox.append(df1[2][0])

words.append(df2[0][0])
labels.append(df2[1][0])
bbox.append(df2[2][0])


with open('final_pickle.pkl', 'wb') as t:
    pickle.dump([words, labels, bbox], t)

df = pd.read_pickle('final_pickle.pkl')

print(df)
