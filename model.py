""" 用于服务器调试模型结构 """

import torch.nn as nn
from torch.nn.parallel import DataParallel

all_feat_cols = [i for i in range(386)]
target_cols = [i for i in range(1)]

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.batch_norm0 = nn.BatchNorm1d(len(all_feat_cols))
        self.dropout0 = nn.Dropout(0.2) # 0.2

        dropout_rate = 0.2 # 0.1>0.2
        hidden_size = 256 # 386>256
        self.dense1 = nn.Linear(len(all_feat_cols), hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.dense2 = nn.Linear(hidden_size+len(all_feat_cols), hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.dense3 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.dense4 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm4 = nn.BatchNorm1d(hidden_size)
        self.dropout4 = nn.Dropout(dropout_rate)

        self.dense5 = nn.Linear(hidden_size+hidden_size, len(target_cols))
        
        
        # ================================
        self.dense41 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm41 = nn.BatchNorm1d(hidden_size)
        self.dropout41 = nn.Dropout(dropout_rate)

        self.dense42 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm42 = nn.BatchNorm1d(hidden_size)
        self.dropout42 = nn.Dropout(dropout_rate)

        self.dense6 = nn.Linear(5*hidden_size, len(target_cols))
        # ================================

        self.Relu = nn.ReLU(inplace=True)
        self.PReLU = nn.PReLU()
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        # self.GeLU = nn.GELU()
        self.RReLU = nn.RReLU()

    def forward(self, x):
        x = self.batch_norm0(x)
        x = self.dropout0(x)

        x1 = self.dense1(x)
        x1 = self.batch_norm1(x1)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x1 = self.LeakyReLU(x1)
        x1 = self.dropout1(x1)

        x = torch.cat([x, x1], 1)

        x2 = self.dense2(x)
        x2 = self.batch_norm2(x2)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x2 = self.LeakyReLU(x2)
        x2 = self.dropout2(x2)

        x = torch.cat([x1, x2], 1)

        x3 = self.dense3(x)
        x3 = self.batch_norm3(x3)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x3 = self.LeakyReLU(x3)
        x3 = self.dropout3(x3)

        x = torch.cat([x2, x3], 1)

        x4 = self.dense4(x)
        x4 = self.batch_norm4(x4)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x4 = self.LeakyReLU(x4)
        x4 = self.dropout4(x4)


        x = torch.cat([x3, x4], 1)



        # # my code
        # x41 = self.dense41(x)
        # x41 = self.batch_norm41(x41)
        # x41 = self.LeakyReLU(x41)
        # x41 = self.dropout41(x41) 
        # x = torch.cat([x4, x41], 1)
        # # my code
        # x42 = self.dense42(x)
        # x42 = self.batch_norm42(x42)
        # x42 = self.LeakyReLU(x42)
        # x42 = self.dropout42(x42) 
        # x = torch.cat([x41, x42], 1)

        # x = self.dense5(x)

        x = torch.cat([x1, x2, x3, x4], 1)
        x = self.dense6(x)

        
        x = x.squeeze()
        
        return x