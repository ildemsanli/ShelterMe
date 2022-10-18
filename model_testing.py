#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 13:09:26 2022

@author: ildem
"""

#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
import pickle


#Prep data to use as rooms dataset 
rooms=pd.read_csv("paris_clean.csv")

rooms.head

rooms.columns           

rooms=rooms[['id',
       'neighbourhood_cleansed', 'latitude', 'longitude', 'room_type',
       'beds', 'kitchen', 'pet', 'bathrooms'
       ]]

#Rename coluns according to data model 
dict_cols={'id':'room_id', 'neighbourhood_cleansed':'neighbourhood', 'bathrooms':'bathroom'}

rooms.rename(columns=dict_cols, inplace=True)

rooms.shape #(49215, 9)



#Prep data to use as guests dataset 
guests=pd.read_csv("user_info.csv")

guests.columns

guests=guests[['participant_id',
       'age', 'num_rejections',
       'Sex']]

#Rename columns according to data model 
cols={'participant_id':'guest_id', 'num_rejections':'num_dependants', 'Sex':'sex'}

guests.rename(columns=cols, inplace=True)

#Adjust num_dependants
guests['num_dependants']=np.where(guests['num_dependants']>4, 0, guests['num_dependants'] )

guests.shape #(9519, 4)

#Check for null values 
rooms.isna().sum() #0

guests.isna().sum()

"""
guest_id           0
age               13
num_dependants     0
sex                0

"""
guests.dropna(inplace=True)

guests.shape #(9506, 4)


#Take 40000 rooms and match with 8000 * 5 guests

rooms_40=rooms.head(40000)

guests_8=guests.head(8000)

guests_40 = pd.concat([guests_8]*5, ignore_index=True)


rooms_40.pet.value_counts() #0    39977, 1       23
#Pets allowed only in 23 rooms, adjust this to allows pets in 2/5th of rooms
rooms_40.loc[0:16000, ['pet']]=[1]
#0    23978, 1    16022

#Add a column with info on guests' pets - random list
has_pet=[1, 0, 0, 1, 0]*8000

#Add is_pet to df
guests_40['has_pet']=has_pet

guests_40['has_pet'].value_counts()
#0    24000, 1    16000

rooms_40.to_csv('rooms_df')

guests_40.to_csv('guests_df')

#Merge rooms and guests
merged_df=pd.concat([rooms_40, guests_40], axis=1)


#Add a target column
merged_df['target']=np.where((merged_df['pet']>=merged_df['has_pet']) & (merged_df['num_dependants']< merged_df['beds']), 1, 0)

merged_df.target.value_counts()
#0    21657, 1    18343


#Encode the categorical columns

merged_df[['neighbourhood','sex','room_type']]= merged_df[['neighbourhood','sex','room_type']].apply(LabelEncoder().fit_transform)


merged_df.columns
""" ['room_id', 'neighbourhood', 'latitude', 'longitude', 'room_type',
       'beds', 'kitchen', 'pet', 'bathroom', 'guest_id', 'age',
       'num_dependants', 'sex', 'has_pet', 'target']
"""

merged_df.isna().sum() #0



#Split x and y 

x=merged_df[['neighbourhood', 'room_type', 'beds',
       'kitchen', 'pet', 'bathroom', 'age', 'num_dependants',
       'sex', 'has_pet']]

y=merged_df[['target']]

# Train-test split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, 
                                                        random_state=42)


model=ExtraTreesClassifier(max_depth=50, n_estimators=3000).fit(x_train, y_train)
y_pred=model.predict(x_test)
acc_train=model.score(x_train, y_train)
acc_test=model.score(x_test, y_test)

####confusion matrix
matrix=metrics.confusion_matrix(y_test, y_pred)

####precision score
p=metrics.precision_score(y_test, y_pred, average='weighted')

####recall score
r=metrics.recall_score(y_test, y_pred, average='weighted')

####f1 score
f1=metrics.f1_score(y_test, y_pred, average='weighted')



print('Accuracy on train:', round(acc_train, 2))
print('\nAccuracy on test:', round(acc_test, 2))
print('\nConfusion matrix\n', matrix)
print('\nPrecision score:', round(p, 2))
print('\nRecall score:', round(r, 2))
print('\nF1 score:', round(f1, 2))


#Save model 

pickle.dump(model, open('shelter_model.p', 'wb'))

loaded_model = pickle.load(open('shelter_model.p', 'rb'))
result = loaded_model.score(x_test, y_test)
print(result)

user_input=[[5, 1, 3, 1, 1, 1, 23, 0, 0, 1]]
print(model.predict(user_input))

print(loaded_model.predict(user_input))
