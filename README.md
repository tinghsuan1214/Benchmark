# A Training-free Neural Architecture Search for Electricity Load Prediction
We present a NAS benchmark that incorporates a hierarchical search space, which combines cell-based and chain-structured architectures. This design is essential to NAS, defining a total of 2,600 possible architectures, and offers a comprehensive benchmark for NAS applications in smart grids (SG).

This repository provides a benchmark dataset and code for evaluating neural network architectures, specifically in the context of short-term load forecasting.

Search Space 1 consists of architectures with either 2 or 3 nodes. For single cells, it includes 100 architectures with 2 nodes and 2,500 architectures with 3 nodes, resulting in a total of 2,600 single-cell architectures.

![image](https://github.com/tinghsuan1214/Benchmark/blob/main/Figure/the_def_space_page.jpg)
<!-- ## Preparation and Download
To set up your environment and start training on the ISET dataset, follow these steps:
1. Create your Conda environment:
   Install Anaconda or Miniconda on your system.
   Download the env.yaml file provided in this repository.
   Run the following command to create the virtual environment.
```
conda env create -f env.yaml
```
2. Activate the virtual environment.
```
conda activate <environment_name>
```
3. Download the ISET Dataset
   You can download the dataset from the link directly from the drive folder here.
   https://drive.google.com/drive/folders/1e633nQBwTdWP2EkYC2D-zKKId8hh7DIK
   

<!-- 4. Pre-trained model
   You can download the dataset from the link directly from the drive folder here.
   https://drive.google.com/drive/folders/1vbDJNpT2Ie0V3D8G6Pmj1d02cxm4t_BU?usp=drive_link -->
   
## How to Use NAS-Bench-STLF
### Download Benchmark CSV

Performance indicators of the corresponding architecture can be obtained through table lookup.

### Architecture string explanation

The string:

`|lstm_1~0|+|lstm_3~0|lstm_2~1|+|none~0|skip_connect~1||lstm_3~2|[dropout->linear->relu->dropout->linear]`
represents a neural network architecture that combines both cell-based and chain-structured designs. The detailed explanation of this architecture is as follows:

1. Cell-based Structure
```
node-0: The input tensor.
node-1: lstm_1(node-0) represents a single-layer LSTM applied to node-0.
node-2: lstm_3(node-0) + lstm_2(node-1) represents a three-layer LSTM applied to node-0, combined with a two-layer LSTM applied to node-1.
node-3: skip-connect(node-0) + conv-3x3(node-1) + skip-connect(node-2) represents a skip connection from node-0 to node-3, along with a convolutional operation on node-1 and another skip connection from node-2.
```
2. Chain-structured
```
[dropout->linear->relu->dropout->linear]
```
represents the fully connected layer structure. The operations are performed in the following order: Dropout, Linear, ReLU, another Dropout, and finally, Linear.

This model combines the Cell-based structure and Chain-structured design to handle multi-layered time series data, providing output predictions. Below is an example of how this model is represented programmatically, using two `nn.ModuleList` objects:

```
InferCell(
  info :: nodes=4, inC=1, outC=64, [1<-(I0-L0) | 2<-(I0-L1,I1-L2) | 3<-(I0-L3,I1-L4,I2-L5)]
  (cells_structure): ModuleList(
    (0): LSTM(
      (lstm): LSTM(1, 64, batch_first=True)
    )
    (1): LSTM(
      (lstm): LSTM(1, 64, num_layers=3, batch_first=True)
    )
    (2): LSTM(
      (lstm): LSTM(64, 64, num_layers=2, batch_first=True)
    )
    (3): Zero(C_in=1, C_out=64, stride=1)
    (4): Identity()
    (5): LSTM(
      (lstm): LSTM(64, 64, num_layers=3, batch_first=True)
    )
  )
  (fc_layers_structure): ModuleList(
    (0): Dropout(p=0.1, inplace=False)
    (1): Linear(in_features=3072, out_features=1536, bias=True)
    (2): ReLU()
    (3): Dropout(p=0.1, inplace=False)
    (4): Linear(in_features=1536, out_features=1, bias=True)
  )
)
```
<!-- ## Citation
If you find that NAS-Bench-STLF helps your research, please consider citing it:
```
@inproceedings{XXX,
  title     = {A Training-free Neural Architecture Search for Electricity Load Prediction},
  author    = {Pei, Ting-Hsuan, Gagan Vishwakarma and Tsai, Chun-Wei},
  booktitle = {...},
  url       = {https://...},
  year      = {2024}
}
``` -->
