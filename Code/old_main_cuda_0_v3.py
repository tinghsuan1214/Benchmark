# -----------------------------------------------------------------
# Programming Language: Python
# Author: Pei Ting-Hsuan
# Purpose: Generating Model_Type_Benchmark
# -----------------------------------------------------------------

# Standard Library Imports
import os
import sys
import time
import random
import shutil

# Related Third-Party Library Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from thop import profile, clever_format

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.modules.loss import _Loss

# Local Application/Library Imports
from genotype_list import ArchitectureDecoder
# from generate_arch import GenerateArchitecture
from cell_operations import CNN_OPS
from cell_operations import LSTM_OPS
from cell_operations import BiLSTM_OPS
from cell_operations import GRU_OPS
from cell_operations import BiGRU_OPS
from cell_operations import TCN_OPS
from cell_operations import Nas_Bench_201_OPS


#--------------------------------------------------------- Input -----------------------------------------------------------

directory_path = sys.argv[1]
log_file_name = sys.argv[2]
model_type = sys.argv[3]
type = str(sys.argv[4])
data_size = int(sys.argv[5]) # numbers of user
window_size = int(sys.argv[6])
batch_size = int(sys.argv[7])
epochs = int(sys.argv[8])
patience = int(sys.argv[9]) # early stopping
user_index = int(sys.argv[10])
# scheduler_step_size = int(sys.argv[10])
# scheduler_gamma = float(sys.argv[11])
num_linear_layer = int(sys.argv[11])
c_in = int(sys.argv[12]) # Input Channels 每個時間點上的特徵數量
c_out = int(sys.argv[13]) # Output Channels 網路產生的特徵數量
num_cells = int(sys.argv[14]) # 設置 cell 的數量
root_dir = sys.argv[15]
all_arch_str_file_path = sys.argv[16]

if model_type == 'CNN':
    selected_ops = CNN_OPS
elif model_type == 'LSTM':
    selected_ops = LSTM_OPS
elif model_type == 'LSTM_nodes3':
    selected_ops = LSTM_OPS
elif model_type == 'BiLSTM':
    selected_ops = BiLSTM_OPS
elif model_type == 'GRU':
    selected_ops = GRU_OPS
elif model_type == 'BiGRU':
    selected_ops = BiGRU_OPS
elif model_type == 'TCN':
    selected_ops = TCN_OPS
elif model_type == 'Nas_Bench_201':
    selected_ops = Nas_Bench_201_OPS
else:
    raise ValueError(f"Unsupported model type: {model_type}")

#------------------------------------------------------ Load dataset -------------------------------------------------------

train_data = np.load('/mnt/4TB/tinghsuan/Model_Type_Benchmark/data/paper_ISET_train_mean.npy') # (21408, 920, 2)
data_train = pd.DataFrame(train_data[..., 0]).transpose()[:data_size] # (data_size, 21408)

valid_data = np.load('/mnt/4TB/tinghsuan/Model_Type_Benchmark/data/paper_ISET_valid_mean.npy') # (2880, 920, 2)
data_valid = pd.DataFrame(valid_data[..., 0]).transpose()[:data_size] # (data_size, 2880)

test_data = np.load('/mnt/4TB/tinghsuan/Model_Type_Benchmark/data/paper_ISET_test_mean.npy') # (1440, 920, 2)
data_test = pd.DataFrame(test_data[..., 0]).transpose()[:data_size] # (data_size, 1440)

#------------------------------------------------------- RMSELoss ----------------------------------------------------------

class RMSELoss(_Loss):

  __constants__ = ['reduction']

  def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
    super(RMSELoss, self).__init__(size_average, reduce, reduction)

  def forward(self, input: Tensor, target: Tensor) -> Tensor:
    mse_loss = F.mse_loss(input, target, reduction=self.reduction)
    rmse_loss = torch.sqrt(mse_loss)
    return rmse_loss
  
#---------------------------------------------------- Custom Dataset -------------------------------------------------------- 

class  CustomDataset(Dataset):
    
  def __init__(self, data, window_size=window_size): # ISET logs every half hour (48 times a day)
    self.data = data
    self.window_size = window_size
    self.row_index = 0  
    self.column_index = 0  

  def __getitem__(self, index):
    if self.column_index + self.window_size >= (self.data.shape[1]):
      self.column_index = 0
      self.row_index += 1
        
    data_list = [self.data.iloc[self.row_index, self.column_index:self.column_index + self.window_size].values]
    data_array = np.array(data_list)  
    x = torch.tensor(data_array)  
    y = torch.tensor(self.data.iloc[self.row_index, self.column_index + self.window_size])
    
    self.column_index += 1

    return x, y
  
  def __len__(self):
    return data_size * (self.data.shape[1] - self.window_size)

#------------------------------------------------------- InferCell -----------------------------------------------------------
    
class InferCell(nn.Module):

  def __init__(self, genotype, linear_layer_str, C_in, C_out, stride, num_cells, window_size, p_drop=0.1):
    super(InferCell, self).__init__()
    self.cells_structure  = nn.ModuleList() 
    self.node_IN = []
    self.node_IX = []
    self.genotype = genotype
    self.fc_layers_structure = nn.ModuleList()
    self.num_cells = num_cells
    self.dropout_p = p_drop
    self.window_size = window_size

    for _ in range(self.num_cells):
      for i in range(1, len(genotype) + 1):
        node_info = genotype[i - 1]
        cur_index = []
        cur_innod = []
        for (op_name, op_in) in node_info:
          if op_in == 0:
              layer = selected_ops[op_name](C_in, C_out, stride, True, True)
          else:
              layer = selected_ops[op_name](C_out, C_out, 1, True, True)
          cur_index.append(len(self.cells_structure))
          cur_innod.append(op_in)
          self.cells_structure.append(layer)
        self.node_IX.append(cur_index)
        self.node_IN.append(cur_innod)

    self.nodes = len(genotype) + 1
    self.in_dim = C_in
    self.out_dim = C_out

    linear_sizes = [C_out * window_size]
    n_lin = linear_layer_str.count('linear')
    for i in range(1, n_lin):
        linear_sizes.append(linear_sizes[-1] // 2)
    linear_sizes.append(1)

    self._create_linear_layers(linear_layer_str, linear_sizes)

  def _create_linear_layers(self, linear_layer_str, linear_sizes):
      linear_ops = linear_layer_str.split('->')
      for op in linear_ops:
          if op == "dropout":
              self.fc_layers_structure.append(nn.Dropout(self.dropout_p))
          elif op == "linear":
              self.fc_layers_structure.append(nn.Linear(linear_sizes.pop(0), linear_sizes[0]))
          elif op == "relu":
              self.fc_layers_structure.append(nn.ReLU())

  def extra_repr(self):
      string = 'info :: nodes={nodes}, inC={in_dim}, outC={out_dim}'.format(**self.__dict__)
      laystr = []
      for i, (node_layers, node_innods) in enumerate(zip(self.node_IX, self.node_IN)):
          y = ['I{:}-L{:}'.format(_ii, _il) for _il, _ii in zip(node_layers, node_innods)]
          x = '{:}<-({:})'.format(i + 1, ','.join(y))
          laystr.append(x)
      genotype_str = '|'.join(['|'.join(['{:}~{:}'.format(op_name, op_in) for (op_name, op_in) in node]) for node in self.genotype])
      linear_strs = []
      for i, layer in enumerate(self.fc_layers_structure):
          if isinstance(layer, nn.Linear):
              linear_strs.append("linear")
          elif isinstance(layer, nn.ReLU):
              linear_strs.append("relu")
          elif isinstance(layer, nn.Dropout):
              linear_strs.append("dropout")
      linear_str = '->'.join(linear_strs)
      return string + ', [{:}]'.format(' | '.join(laystr))

  def forward(self, inputs):
      nodes = [inputs]
      for i, (node_layers, node_innods) in enumerate(zip(self.node_IX, self.node_IN)):
          node_feature = [self.cells_structure[_il](nodes[_ii]) for _il, _ii in zip(node_layers, node_innods)]
          node_feature = sum(node_feature)
          nodes.append(node_feature)

      out = nodes[-1].reshape(nodes[-1].size(0), -1)

      for layer in self.fc_layers_structure:
        out = layer(out)

      return out

  def get_linear_str(self):
      linear_strs = []
      for i, layer in enumerate(self.fc_layers_structure):
          if isinstance(layer, nn.Linear):
              linear_strs.append("linear")
          elif isinstance(layer, nn.ReLU):
              linear_strs.append("relu")
          elif isinstance(layer, nn.Dropout):
              linear_strs.append("dropout")
      linear_str = '->'.join(linear_strs)
      return linear_str

#------------------------------------------------------- Training -----------------------------------------------------------

log_file_path = os.path.join(directory_path, f'{log_file_name}.txt')
os.makedirs(directory_path, exist_ok=True)
with open(log_file_path, 'w', buffering=1) as log_file:
  original_stdout = sys.stdout
  sys.stdout = log_file
  start_time = time.time()
  
  print("--------------------Training--------------------")
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  def read_and_update_architecture(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    arch_str = lines[0].strip()  # 讀取首行，即當前的神經網络架構字符串
    remaining_lines = lines[1:]  # 保存文件的剩餘部分
    # 更新文件
  #   with open(file_path, 'w') as file:
  #       file.writelines(remaining_lines)
  #   return arch_str
  # arch_str = read_and_update_architecture('/mnt/4TB/tinghsuan/Model_Type_Benchmark/arch/best.txt')
    # 提取 [] 中的字串
    start = arch_str.find('[')
    end = arch_str.find(']')
    linear_layer = arch_str[start+1:end] if start != -1 and end != -1 else ''
    
    # 去除 [] 中的部分，保留其餘部分
    arch_str = arch_str[:start].strip() + arch_str[end+1:].strip() if start != -1 and end != -1 else arch_str
    
    # 更新文件
    with open(file_path, 'w') as file:
        file.writelines(remaining_lines)
    
    return arch_str, linear_layer

  arch_str, linear_layer = read_and_update_architecture('/mnt/4TB/tinghsuan/Model_Type_Benchmark/arch/bench_v1_0.txt')

  # 輸出結果檢查
  # print("arch_str:", arch_str)
  # print("linear_layer:", linear_layer)

  # generator = GenerateArchitecture(model_type, 3)
  # arch_str = generator.get_architecture_string()
  
  # def is_arch_str_exist(file_path, arch_str):
  #   if not os.path.exists(file_path):
  #       open(file_path, 'a').close()

  #   with open(file_path, 'r') as file:
  #       for line in file:
  #           if arch_str in line.strip():
  #               return True
  #   return False

  # all_arch_str_file_path = all_arch_str_file_path

  # generator = GenerateArchitecture(model_type, 3)

  # MAX_TRIES = 100  
  # tries = 0
  # unique_arch_str_found = False
  # while not unique_arch_str_found:
  #   arch_str = generator.get_architecture_string()
  #   if not is_arch_str_exist(all_arch_str_file_path, arch_str):
  #     unique_arch_str_found = True
  #   else: 
  #     tries += 1
  #     if tries >= MAX_TRIES:
  #       # print("Unable to find a unique architecture string.")
  #       sys.exit()

  # if unique_arch_str_found:
    # print("A unique architecture string has been found: ", arch_str)
        
  genotype = ArchitectureDecoder.str2lists(arch_str)
  model = InferCell(genotype, linear_layer, C_in=c_in, C_out=c_out, stride=1, num_cells=num_cells, window_size=window_size).to(device).double()
  linear_str = model.get_linear_str()
  if num_cells == 2:
    print("arch_str :: {:}-{:}[{:}]".format(arch_str, arch_str, linear_str))
  else:
    print("arch_str :: {:}[{:}]".format(arch_str, linear_str))
  print("model :: {:}\n{:}".format(type, model))
  print("tr_params :: epochs = {:}, patience = {:}, batch_size = {:}, window_size = {:}, data_size = {:}".format(epochs, patience, batch_size, window_size, data_size))
  
  typical_input = torch.randn(1, c_in, window_size).double().to(device) 
  flops, params = profile(model, inputs=(typical_input, ), verbose=False)
  flops, params = clever_format([flops, params], "%.3f")
  print(f"Model FLOPs: {flops}, Model Params: {params}")

  criterion = RMSELoss().double()
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
  total_steps = 15
  # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
  print("learning rate scheduling :: (optimizer, mode='min', factor=0.1, patience=5, verbose=True)")

  train_dataset = CustomDataset(data_train)
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
  print("total_samples :: train_dataset = {:}, train_dataloader = {:}".format(len(train_dataset), len(train_dataloader.dataset)))
  # sample_x, sample_y = train_dataset[0]
  # print("Input (x) dimensions:", sample_x.size())
  # print("Output (y) dimensions:", sample_y.size())

  valid_dataset = CustomDataset(data_valid)
  valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
  print("total_samples :: valid_dataset = {:}, valid_dataloader = {:}".format(len(valid_dataset), len(valid_dataloader.dataset)))

  loss_sum = 0
  it_count = 0
  patience_counter = 0
  best_loss = float('inf')
  early_stopping_lr_threshold = 1e-7  # 設定學習率早停閾值

  for epoch in range(epochs):
    model.train()
    loss_sum=0
    it_count = 0
    for (x, y) in train_dataloader:
      optimizer.zero_grad()
      x = x.to(device)
      y = y.to(device)
      y_pred = model(x).double()
      y_pred = y_pred.squeeze(1)
      loss = criterion(y_pred, y.double())
      loss_sum += loss.item()
      loss.backward() 
      optimizer.step() 
      it_count += 1 
  # scheduler.step()

    average_loss = loss_sum / it_count
    train_dataloader.dataset.row_index = 0
    train_dataloader.dataset.column_index = 0

    model.eval()
    val_loss_sum = 0
    val_it_count = 0

    with torch.no_grad():
      for (x_val, y_val) in valid_dataloader:
        x_val = x_val.to(device)
        y_val = y_val.to(device)
        y_val_pred = model(x_val)
        y_val_pred = y_val_pred.squeeze(1)
        val_loss = criterion(y_val_pred, y_val.float())
        val_loss_sum += val_loss.item()
        val_it_count += 1

    average_val_loss = val_loss_sum / val_it_count
    if epoch > total_steps:
      scheduler.step(average_val_loss)
    valid_dataloader.dataset.row_index = 0
    valid_dataloader.dataset.column_index = 0
    
    # 檢查學習率是否低於早停閾值
    current_lr = optimizer.param_groups[0]['lr']
    if current_lr < early_stopping_lr_threshold:
      print("Early stopping triggered due to learning rate below threshold.")
      break
    
    epoch_time = time.time() - start_time
    hours, remainder = divmod(epoch_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch :: {epoch + 1:2} || Loss: {average_loss:10.8f} || it_count: {it_count} ||'
      f' Val Loss: {average_val_loss:10.8f} || Val it_count: {val_it_count} || Current Learning Rate: {current_lr} || Time: {int(hours):02}:{int(minutes):02}:{seconds:.2f}')

    if average_val_loss < best_loss:
      best_loss = average_val_loss
      patience_counter = 0
      torch.save(model, f'{directory_path}/{log_file_name}.pt')
    else:
      patience_counter += 1

    if patience_counter >= patience:
      print("Early stopping triggered due to patience exceeded.")
      break

  run_time = time.time() - start_time
  hours, remainder = divmod(run_time, 3600)
  minutes, seconds = divmod(remainder, 60)
  print(f'Done Total time: {int(hours):02}:{int(minutes):02}:{seconds:.2f}')
  print(f'best_loss: {best_loss}')

  #------------------------------------------------------ Testing --------------------------------------------------------------

  print("\n--------------------Testing--------------------\n")
  test_dataset = CustomDataset(data_test)
  test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
  print("total_samples :: test_dataset = {:}, test_dataloader = {:}".format(len(test_dataset), len(test_dataloader.dataset)))
  model_path = f'{directory_path}/{log_file_name}.pt'
  model = torch.load(model_path, map_location=device)
  model.eval()

  it_count = 0
  loss_sum = 0
  predictions = []
  y_list = []
  criterion = nn.MSELoss().double()

  start_time = time.time()
  with torch.no_grad():
    for x, y in test_dataloader:
      x = x.to(device)
      y = y.to(device)
      y_list.append(y.double().cpu()) 
      y_pred = model(x)
      y_pred = y_pred.squeeze(1) 
      predictions.extend(y_pred.cpu().numpy().tolist())
      loss = criterion(y_pred.double(), y.double())
      loss_sum += loss.item()
      it_count += 1 

  test_dataloader.dataset.row_index = 0
  test_dataloader.dataset.column_index = 0
  average_loss = loss_sum / it_count
  end_time = time.time() - start_time
  hours, remainder = divmod(end_time, 3600)
  minutes, seconds = divmod(remainder, 60)
  print(f'Epoch :: {1:2} || Loss: {average_loss:10.8f} || it_count: {it_count} || Time: {int(hours):02}:{int(minutes):02}:{seconds:.2f}') 

  #------------------------------------------------------- Drawing ------------------------------------------------------------

  y_list_flat = [item.cpu() for sublist in y_list for item in sublist]

  predictions = np.array(predictions).astype(np.float32)
  print(predictions.shape)
  pred = predictions.reshape(data_size, -1)
  print(pred.shape)
  y_list = np.array(y_list_flat).astype(np.float32)
  print(y_list.shape)
  y = y_list.reshape(data_size, -1)
  print(y.shape)

  mae = mean_absolute_error(y_list, predictions)
  mse = np.mean((predictions - y_list) ** 2)

  sub_arr_square = np.square(pred - y)
  sub_arr_square_row_mean = np.mean(sub_arr_square, axis=1)
  sub_arr_square_row_mean_sqrt = np.sqrt(sub_arr_square_row_mean)
  rmse = np.mean(sub_arr_square_row_mean_sqrt)

  ymin, ymax = np.min(y,axis=1), np.max(y,axis=1)
  print(ymin.shape)
  print(ymax.shape)
  nrmse = np.mean(sub_arr_square_row_mean_sqrt / (ymax - ymin))

  print("MAE: ", mae)
  print("MSE: ", mse)
  print("RMSE: ", rmse)
  print("NRMSE: ", nrmse)

  user = data_test.iloc[user_index, window_size : 336 + window_size].values 
  preds = pred[user_index, : 336]

  plt.figure(dpi = 120, figsize=(20, 5))
  plt.plot(user, label='loads')
  plt.plot(preds, label = 'predictions')
  plt.title('User {:} Load and Prediction'.format(user_index + 1))
  plt.xlabel('Timestep')
  plt.ylabel('Load')
  plt.legend()
  plt.savefig(f'{directory_path}/{log_file_name}.png')
  plt.close()
  sys.stdout = original_stdout

  target_directory = root_dir
  os.makedirs(target_directory, exist_ok=True)
  target_file_path = os.path.join(target_directory, f'{log_file_name}.txt')
  shutil.copy(log_file_path, target_file_path)