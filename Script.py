# Feature learning 8.1 

# import modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.notebook import tqdm

#-----------------------------------------------------------------------
def feature_extraction(df): 
  rows=[]
  for id,row in tqdm(df.iterrows(), total=df.shape[0]): 
    row_data = dict(
        year = row.date.year,
        month = row.date.month,
        day = row.date.day,
        close = row.close
    )
    rows.append(row_data)
  return pd.DataFrame(rows)
#-----------------------------------------------------------------------
def split_data(df): 
  train_size = int(len(df)*0.75)
  test_size = int(len(df)*0.15)
  train_df,test_df = df[:train_size], df[train_size:train_size+test_size]
  valid_df = df[train_size+test_size:]
  return train_df,test_df,valid_df


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#-----------------------------------------------------------------------
# transform data
def split_transform(df):

  train_df,test_df,valid_df = split_data(df)

  print("train shape :",train_df.shape)
  print("test shape :",test_df.shape)
  print("validation data ",valid_df.shape)
  scaler = MinMaxScaler(feature_range=(-1,1))

  train_df = pd.DataFrame(
      scaler.fit_transform(train_df),
      index=train_df.index,
      columns = train_df.columns
  )
  test_df = pd.DataFrame(
      scaler.fit_transform(test_df),
      index=test_df.index, 
      columns = test_df.columns
  )
  valid_df = pd.DataFrame(
      scaler.fit_transform(valid_df),
      index=valid_df.index, 
      columns = valid_df.columns
  )
  return train_df,test_df,valid_df, scaler # we'll use scaler later when unscaling

#-----------------------------------------------------------------------
def create_sequences(input_data,sequence_len): 
  seqs = []
  for i in tqdm(range(len(input_data)-sequence_len)):
    seq = input_data[i:i+sequence_len] # sequencing data
    label_position= i+sequence_len # keep track of current position
    label= input_data.iloc[label_position]["close"]
    seqs.append((seq,label))
  return seqs

#----------------------------------------------------------------------
# Explore Prediction
#-----------------------------------------------------------------------
import seaborn as sns 
def explore_predictions(predict,test_targets):
    plt.figure(figsize=(12,9))
    sns.set_style("darkgrid")
    # Plot the predictions against the actual values
    plt.plot(test_targets, label='actual')
    plt.plot(predict, label='prediction',color="red")
    plt.legend()
    plt.show()