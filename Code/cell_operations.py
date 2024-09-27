import torch.nn as nn
import torch
import torch.nn.functional as F

CNN_OPS = {

  'conv_1' : lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, 1, stride, 0, 1, affine, track_running_stats),
  'conv_2' : lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, 2, stride, 'same', 1, affine, track_running_stats),
  'conv_3' : lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, 3, stride, 1, 1, affine, track_running_stats),
  'conv_4' : lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, 4, stride, 'same', 1, affine, track_running_stats),
  'conv_5' : lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, 5, stride, 2, 1, affine, track_running_stats),
  'conv_6' : lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, 6, stride, 'same', 1, affine, track_running_stats),
  
  'avg_pool_3' : lambda C_in, C_out, stride, affine, track_running_stats: POOLING(C_in, C_out, 3, stride, 1, 'avg', affine, track_running_stats),
  'max_pool_3' : lambda C_in, C_out, stride, affine, track_running_stats: POOLING(C_in, C_out, 3, stride, 1, 'max', affine, track_running_stats),
  'avg_pool_5' : lambda C_in, C_out, stride, affine, track_running_stats: POOLING(C_in, C_out, 5, stride, 2, 'avg', affine, track_running_stats),
  'max_pool_5' : lambda C_in, C_out, stride, affine, track_running_stats: POOLING(C_in, C_out, 5, stride, 2, 'max', affine, track_running_stats),
  'avg_pool_7' : lambda C_in, C_out, stride, affine, track_running_stats: POOLING(C_in, C_out, 7, stride, 3, 'avg', affine, track_running_stats),
  'max_pool_7' : lambda C_in, C_out, stride, affine, track_running_stats: POOLING(C_in, C_out, 7, stride, 3, 'max', affine, track_running_stats),
  'none'         : lambda C_in, C_out, stride, affine, track_running_stats: Zero(C_in, C_out, stride),
  'skip_connect' : lambda C_in, C_out, stride, affine, track_running_stats: Identity() if stride == 1 and C_in == C_out else FactorizedReduce(C_in, C_out, stride, affine, track_running_stats)

}

LSTM_OPS = {
  
  'lstm_1': lambda input_dim, hidden_dim, stride, affine, track_running_stats: LSTM(input_dim, hidden_dim, 1, 0.0),
  'lstm_2': lambda input_dim, hidden_dim, stride, affine, track_running_stats: LSTM(input_dim, hidden_dim, 2, 0.0),
  'lstm_3': lambda input_dim, hidden_dim, stride, affine, track_running_stats: LSTM(input_dim, hidden_dim, 3, 0.0),
  'lstm_4': lambda input_dim, hidden_dim, stride, affine, track_running_stats: LSTM(input_dim, hidden_dim, 4, 0.0),
  'lstm_5': lambda input_dim, hidden_dim, stride, affine, track_running_stats: LSTM(input_dim, hidden_dim, 5, 0.0),
  'none'         : lambda C_in, C_out, stride, affine, track_running_stats: Zero(C_in, C_out, stride),
  'skip_connect' : lambda C_in, C_out, stride, affine, track_running_stats: Identity() if stride == 1 and C_in == C_out else FactorizedReduce(C_in, C_out, stride, affine, track_running_stats),
  # 'attention': lambda input_dim, hidden_dim, stride, affine, track_running_stats: SelfAttention(input_dim, hidden_dim),
}

BiLSTM_OPS = {
  
  'bilstm_1': lambda input_dim, hidden_dim, stride, affine, track_running_stats: BiLSTM(input_dim, hidden_dim//2, 1, 0.0),
  'bilstm_2': lambda input_dim, hidden_dim, stride, affine, track_running_stats: BiLSTM(input_dim, hidden_dim//2, 2, 0.0),
  'bilstm_3': lambda input_dim, hidden_dim, stride, affine, track_running_stats: BiLSTM(input_dim, hidden_dim//2, 3, 0.0),
  'bilstm_4': lambda input_dim, hidden_dim, stride, affine, track_running_stats: BiLSTM(input_dim, hidden_dim//2, 4, 0.0),
  'bilstm_5': lambda input_dim, hidden_dim, stride, affine, track_running_stats: BiLSTM(input_dim, hidden_dim//2, 5, 0.0),
  'bilstm_6': lambda input_dim, hidden_dim, stride, affine, track_running_stats: BiLSTM(input_dim, hidden_dim//2, 6, 0.0),
  'none'         : lambda C_in, C_out, stride, affine, track_running_stats: Zero(C_in, C_out, stride),
  'skip_connect' : lambda C_in, C_out, stride, affine, track_running_stats: Identity() if stride == 1 and C_in == C_out else FactorizedReduce(C_in, C_out, stride, affine, track_running_stats)

}

GRU_OPS = {
  
  'gru_1': lambda input_dim, hidden_dim, stride, affine, track_running_stats: GRU(input_dim, hidden_dim, 1, 0.0),
  'gru_2': lambda input_dim, hidden_dim, stride, affine, track_running_stats: GRU(input_dim, hidden_dim, 2, 0.0),
  'gru_3': lambda input_dim, hidden_dim, stride, affine, track_running_stats: GRU(input_dim, hidden_dim, 3, 0.0),
  'gru_4': lambda input_dim, hidden_dim, stride, affine, track_running_stats: GRU(input_dim, hidden_dim, 4, 0.0),
  'gru_5': lambda input_dim, hidden_dim, stride, affine, track_running_stats: GRU(input_dim, hidden_dim, 5, 0.0),
  'gru_6': lambda input_dim, hidden_dim, stride, affine, track_running_stats: GRU(input_dim, hidden_dim, 6, 0.0),
  'none'         : lambda C_in, C_out, stride, affine, track_running_stats: Zero(C_in, C_out, stride),
  'skip_connect' : lambda C_in, C_out, stride, affine, track_running_stats: Identity() if stride == 1 and C_in == C_out else FactorizedReduce(C_in, C_out, stride, affine, track_running_stats)

}

BiGRU_OPS = {
  
  'bigru_1': lambda input_dim, hidden_dim, stride, affine, track_running_stats: BiGRU(input_dim, hidden_dim//2, 1, 0.0),
  'bigru_2': lambda input_dim, hidden_dim, stride, affine, track_running_stats: BiGRU(input_dim, hidden_dim//2, 2, 0.0),
  'bigru_3': lambda input_dim, hidden_dim, stride, affine, track_running_stats: BiGRU(input_dim, hidden_dim//2, 3, 0.0),
  'bigru_4': lambda input_dim, hidden_dim, stride, affine, track_running_stats: BiGRU(input_dim, hidden_dim//2, 4, 0.0),
  'bigru_5': lambda input_dim, hidden_dim, stride, affine, track_running_stats: BiGRU(input_dim, hidden_dim//2, 5, 0.0),
  'bigru_6': lambda input_dim, hidden_dim, stride, affine, track_running_stats: BiGRU(input_dim, hidden_dim//2, 6, 0.0),
  'none'         : lambda C_in, C_out, stride, affine, track_running_stats: Zero(C_in, C_out, stride),
  'skip_connect' : lambda C_in, C_out, stride, affine, track_running_stats: Identity() if stride == 1 and C_in == C_out else FactorizedReduce(C_in, C_out, stride, affine, track_running_stats)
}

TCN_OPS = {

  'tcn' : lambda C_in, C_out, stride, affine, track_running_stats: TCN(C_in, C_out, [C_out]*3, 3, 0.2)

}

Nas_Bench_201_OPS = {
 
  'nor_conv_1' : lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, 1, stride, 0, 1, affine, track_running_stats),
  'nor_conv_3' : lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, 3, stride, 1, 1, affine, track_running_stats),
  'avg_pool_3' : lambda C_in, C_out, stride, affine, track_running_stats: POOLING(C_in, C_out, 3, stride, 1, 'avg', affine, track_running_stats),
  'none'         : lambda C_in, C_out, stride, affine, track_running_stats: Zero(C_in, C_out, stride),
  'skip_connect' : lambda C_in, C_out, stride, affine, track_running_stats: Identity() if stride == 1 and C_in == C_out else FactorizedReduce(C_in, C_out, stride, affine, track_running_stats)
}

#----------------------------------------------------  CNN -------------------------------------------------------------

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine, track_running_stats=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
      nn.BatchNorm1d(C_out, affine=affine, track_running_stats=track_running_stats)
    )

  def forward(self, x):
    x = self.op(x)
    return x

class POOLING(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, mode, affine=True, track_running_stats=True):
    super(POOLING, self).__init__()
    if C_in == C_out:
      self.preprocess = None
    else:
      self.preprocess = ReLUConvBN(C_in, C_out, 1, 1, 0, 1, affine, track_running_stats)
    if mode == 'avg'  : self.op = nn.AvgPool1d(kernel_size, stride=stride, padding=padding, count_include_pad=False)
    elif mode == 'max': self.op = nn.MaxPool1d(kernel_size, stride=stride, padding=padding)
    else              : raise ValueError('Invalid mode={:} in POOLING'.format(mode))

  def forward(self, inputs):
    if self.preprocess: x = self.preprocess(inputs)
    else              : x = inputs
    return self.op(x)
  
#----------------------------------------------------  LSTM ------------------------------------------------------------

class LSTM(nn.Module):

  def __init__(self, input_dim, hidden_dim, num_layers, dropout):
    super(LSTM, self).__init__()
    dropout = dropout if num_layers > 1 else 0.0
    self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)

  def forward(self, x):
    x = x.permute(0, 2, 1) 
    # print(x.shape) # torch.Size([256, 48, 1]) (batch_size=256, seq_len=48, input_dim=1)
    lstm_output, _ = self.lstm(x)
    out = lstm_output.permute(0, 2, 1)
    # print(out.shape) # torch.Size([256, 64, 48]) (batch_size=256, hidden_dim=64, seq_len=48)
    return out

# class SelfAttention(nn.Module):
#   def __init__(self, input_dim, hidden_dim, batch_size=256):
#     super(SelfAttention, self).__init__()
#     self.hidden_dim = hidden_dim
#     self.query = nn.Linear(batch_size, input_dim, hidden_dim)
#     self.key = nn.Linear(batch_size, input_dim, hidden_dim)
#     self.value = nn.Linear(batch_size, input_dim, hidden_dim)
#     self.softmax = nn.Softmax(dim=-1)
#     self.batch_size = batch_size

#   def forward(self, x):
#     print(x.shape)
#     if x.shape[0] != self.batch_size:
#       x = x.expand(self.batch_size, -1, -1)  # Expand along batch dimension if necessary
#     print(x.shape) # torch.Size([256, 1, 48])
#     x = x.squeeze(2)  # Squeeze the last dimension if it's 1
#     print(x.shape)
#     query = self.query(x)  # (batch_size, seq_len, hidden_dim)
#     key = self.key(x)      # (batch_size, seq_len, hidden_dim)
#     value = self.value(x)  # (batch_size, seq_len, hidden_dim)

#     scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32))  # scores: (batch_size, seq_len, seq_len)
#     attn_weights = self.softmax(scores)  # attn_weights: (batch_size, seq_len, seq_len)

#     context = torch.matmul(attn_weights, value.transpose(-2, -1))  # context: (batch_size, seq_len, hidden_dim)
#     context = context.transpose(-2, -1)  # Transpose back to (batch_size, hidden_dim, seq_len)
#     return context.unsqueeze(2) 


#---------------------------------------------------  BiLSTM -----------------------------------------------------------

class BiLSTM(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_layers, dropout, bidirectional=True):
    super(BiLSTM, self).__init__()
    self.bidirectional = bidirectional
    self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=self.bidirectional)

  def forward(self, x):
    x = x.permute(0, 2, 1) 
    lstm_output, _ = self.lstm(x)
    return lstm_output.permute(0, 2, 1)

#---------------------------------------------------  GRU ------------------------------------------------------------

class GRU(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_layers, dropout, bidirectional=False):
    super(GRU, self).__init__()
    self.bidirectional = bidirectional
    self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=self.bidirectional)

  def forward(self, x):
    x = x.permute(0, 2, 1) 
    gru_output, _ = self.gru(x)
    out = gru_output.permute(0, 2, 1)
    return out
  
#---------------------------------------------------  BiGRU ------------------------------------------------------------

class BiGRU(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_layers, dropout, bidirectional=True):
    super(BiGRU, self).__init__()
    self.bidirectional = bidirectional
    self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=self.bidirectional)

  def forward(self, x):
    x = x.permute(0, 2, 1) 
    gru_output, _ = self.gru(x)
    out = gru_output.permute(0, 2, 1)
    return out
  
#----------------------------------------------------  TCN -------------------------------------------------------------

# 定義因果卷積層
class CausalConv1d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
    super(CausalConv1d, self).__init__()
    self.padding = (kernel_size - 1) * dilation
    self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

  def forward(self, x):
    out = self.conv1d(x)
    out = out[:, :, :-self.padding]  # 移除多餘的padding
    return out

# 定義TCN殘差區塊
class TCNBlock(nn.Module):
  def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
    super(TCNBlock, self).__init__()
    self.conv1 = CausalConv1d(n_inputs, n_outputs, kernel_size, dilation=dilation)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(dropout)
    self.conv2 = CausalConv1d(n_outputs, n_outputs, kernel_size, dilation=dilation)
    self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
    self.relu = nn.ReLU()
    self.init_weights()

  def init_weights(self):
    self.conv1.conv1d.weight.data.normal_(0, 0.01)
    self.conv2.conv1d.weight.data.normal_(0, 0.01)
    if self.downsample:
        self.downsample.weight.data.normal_(0, 0.01)

  def forward(self, x):
    out = self.relu(self.conv1(x))
    out = self.dropout(out)
    out = self.conv2(out)

    res = x if self.downsample is None else self.downsample(x)
    return self.relu(out + res)

# 定義TCN模型
class TCN(nn.Module):
  def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
    super(TCN, self).__init__()
    layers = []
    num_levels = len(num_channels)
    for i in range(num_levels):
        dilation_size = 2 ** i
        in_channels = input_size if i == 0 else num_channels[i-1]
        out_channels = num_channels[i]
        layers += [TCNBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout)]

    self.tcn = nn.Sequential(*layers)

  def forward(self, x):
    x = x.transpose(1, 2)  # 調整為(batch_size, channels, seq_length)
    x = x.permute(0, 2, 1)
    y = self.tcn(x)
    return y
  
#---------------------------------------------------  other ------------------------------------------------------------

# none
class Zero(nn.Module):

  def __init__(self, C_in, C_out, stride):
    super(Zero, self).__init__()
    self.C_in   = C_in
    self.C_out  = C_out
    self.stride = stride
    self.is_zero = True

  def forward(self, x):
    if self.C_in == self.C_out:
      if self.stride == 1: return x.mul(0.)
      else               : return x[:,:,::self.stride,::self.stride].mul(0.)
    else:
      shape = list(x.shape)
      shape[1] = self.C_out
      zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
      return zeros

  def extra_repr(self):
    return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)

# skip_connect 
class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

# skip_connect
class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, stride, affine, track_running_stats):
    super(FactorizedReduce, self).__init__()
    self.stride = stride
    self.C_in   = C_in  
    self.C_out  = C_out  
    self.relu   = nn.ReLU(inplace=False)
    if stride == 2:
      #assert C_out % 2 == 0, 'C_out : {:}'.format(C_out)
      C_outs = [C_out // 2, C_out - C_out // 2]
      self.convs = nn.ModuleList()
      for i in range(2):
        self.convs.append( nn.Conv1d(C_in, C_outs[i], 1, stride=stride, padding=0, bias=False) )
      self.pad = nn.ConstantPad1d((0, 1), 0)
    elif stride == 1:
      self.conv = nn.Conv1d(C_in, C_out, 1, stride=stride, padding=0, bias=False)
    else:
      raise ValueError('Invalid stride : {:}'.format(stride))
    self.bn = nn.BatchNorm1d(C_out, affine=affine, track_running_stats=track_running_stats)

  def forward(self, x):
    if self.stride == 2:
      x = self.relu(x)
      y = self.pad(x)
      out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:])], dim=1)
    else:
      out = self.conv(x)
    out = self.bn(out)
    return out

  def extra_repr(self):
    return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)

