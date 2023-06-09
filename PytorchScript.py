# import modules
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler



#-----------------------------------------------------------------------
class SequenceDataset(Dataset):
  def __init__(self,sequences): 
    self.sequences = sequences

  def __len__(self):
    return len(self.sequences)

  def __getitem__(self,idx):
    sequence,label = self.sequences[idx]
    return dict(
        sequence=torch.Tensor(sequence.to_numpy()),
        label=torch.tensor(label).float()
    )

#-----------------------------------------------------------------------
class SalesDataset(Dataset):
  def __init__(self,train_sequences,test_sequences,batch_size=1):
    self.train_sequences = train_sequences
    self.test_sequences = test_sequences
    self.batch_size = batch_size

  def setup(self): 
    self.train_dataset = SequenceDataset(self.train_sequences)
    self.test_dataset = SequenceDataset(self.test_sequences)

  def train_dataloader(self): 
    return DataLoader(
        self.train_dataset,
        batch_size=self.batch_size,
    )
  def test_dataloader(self): 
    return DataLoader(
      self.test_dataset, 
      batch_size=self.batch_size,
  )
#-----------------------------------------------------------------------

class SalesPredictions(): 
  def __init__(self,model,criterion,optimizer): 
    self.model = model
    self.criterion = criterion
    self.optimizer = optimizer

  def train_test_rnn(self,num_epochs,train_data,test_data):
    
    self.model.train()
    train_losses=[]
    test_losses=[]
    for epoch in tqdm(range(num_epochs)):
      train_loss = 0.0
      for train_batch in train_data:
          sequences = train_batch["sequence"]
          labels = train_batch["label"]

          self.optimizer.zero_grad()
          outputs = self.model.forward(sequences) 
          loss = self.criterion(outputs, labels)
          loss.backward()
          self.optimizer.step()
          train_loss += loss.item()

      test_loss = 0.0
      with torch.no_grad():
        for test_batch in test_data:
          sequences = test_batch["sequence"]
          labels = test_batch["label"]  
          outputs = self.model.forward(sequences)
          loss = self.criterion(outputs, labels)
          test_loss += loss.item()

      train_loss = train_loss / len(train_data)
      test_loss = test_loss / len(test_data)
      train_losses.append(train_loss)
      test_losses.append(test_loss)
      print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    torch.save(self.model.state_dict(), 'rnn-model.pt') 
    # Plot the training and test loss
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
#-----------------------------------------------------------------------

    
    
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss
    
#-----------------------------------------------------------------------
# FOR MODEL EVALUATION
#-----------------------------------------------------------------------
def model_evaluation(model,input_data):
  predictions_seq=[]
  with torch.no_grad():
    for batch in input_data:
        predictions = model(batch["sequence"])
        pred = torch.flatten(predictions)
        predictions=pred.numpy()
        predictions_seq.append(predictions)
  return predictions_seq
#-----------------------------------------------------------------------
def predictions_preprocess(list_predictions,input_data):
  predictions = pd.Series(data=np.concatenate(list_predictions))
  predictions = predictions.to_frame()

  data_target = input_data['close']
  data_target=data_target[10:].reset_index() # predictions start after the first sequence of len 100
  data_target.drop(["index"],axis=1,inplace=True)
  return predictions,data_target
#-----------------------------------------------------------------------
def descale_transform(predictions_df,target_df,scaler):
  descaler = MinMaxScaler()
  descaler.min_, descaler.scale_ = scaler.min_[-1], scaler.scale_[-1] #get the corresponding values for close (last feature)
  descaled_predictions_df = pd.DataFrame(
      descaler.inverse_transform(predictions_df),
      index=predictions_df.index,
      columns = predictions_df.columns
  )
  descaled_target_df = pd.DataFrame(
      descaler.inverse_transform(target_df),
      index=target_df.index,
      columns = target_df.columns
  )
  return descaled_predictions_df, descaled_target_df
#-----------------------------------------------------------------------
def reformat_data(pred,target,data_origin): 
  data_origin.reset_index(inplace=True)
  pred_descaled, target_descaled = descale_transform(pred,target,Scaler)
  pred_descaled['dates'] = data_origin['date']
  target_descaled['dates'] = data_origin['date']
  pred_descaled['dates'] = pd.to_datetime(pred_descaled['dates'])
  target_descaled['dates'] = pd.to_datetime(target_descaled['dates'])
  target_descaled.set_index('dates',inplace=True)
  pred_descaled.set_index('dates',inplace=True)
  return pred_descaled,target_descaled