import torch  
import torch.nn as nn  
import math  
  
class DynamicGraphGenerate(nn.Module):  
   """  
   Dynamic Graph Generation Module.  
  
   Args:  
      in_channels (int): Input channel size.  
      hidden_channels (int): Hidden channel size.  
      dropout_prob (float): Dropout probability.  
      node_num (int): Number of nodes.  
      reduction (int, optional): Reduction factor. Defaults to 16.  
      alpha (int, optional): Alpha value. Defaults to 3.  
      norm (str, optional): Normalization type. Defaults to 'D-1'.  
   """  
  
   def __init__(self, in_channels: int, hidden_channels: int, dropout_prob: float, node_num: int, reduction: int = 16, alpha: int = 3, norm: str = 'D-1'):  
      super(DynamicGraphGenerate, self).__init__()  
      self.in_channels = in_channels  
      self.hidden_channels = hidden_channels  
      self.node_num = node_num  
      self.reduction = reduction  
      self.alpha = alpha  
      self.norm = norm  
  
      self.start_FC = nn.Linear(in_channels + hidden_channels, hidden_channels)  
      self.avg_pool = nn.AdaptiveAvgPool1d(1)  
      self.fc = nn.Sequential(  
        nn.Linear(node_num, hidden_channels // reduction, bias=False),  
        nn.ReLU(inplace=True),  
        nn.Linear(hidden_channels // reduction, node_num, bias=False),  
        nn.Sigmoid()  
      )  
      self.dropout = nn.Dropout(dropout_prob)  
  
   def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:  
      """  
      Forward pass.  
  
      Args:  
        input (torch.Tensor): Input tensor.  
        hidden (torch.Tensor): Hidden state tensor.  
  
      Returns:  
        torch.Tensor: Normalized adjacency matrix.  
      """  
      x = input  
      batch_size, node_num, hidden_dim = x.shape  
      node_feature = torch.cat([input, hidden], dim=-1)  
      node_feature = node_feature.view(-1, self.in_channels + self.hidden_channels)  
      node_feature = self.start_FC(node_feature)  
      residual = node_feature  
      residual = self.avg_pool(residual)  
      residual = residual.view(-1, hidden_dim, node_num)  
      residual = self.fc(residual)  
      node_feature = torch.mul(residual, node_feature)  
      similarity = torch.matmul(node_feature, node_feature.transpose(0, 1)) / math.sqrt(hidden_dim)  
      if self.norm == 'D-1':  
        adj = F.relu(torch.tanh(self.alpha * similarity))  
        norm_adj = adj / torch.unsqueeze(adj.sum(dim=-1), dim=1)  
      elif self.norm == 'oftmax':  
        adj = F.softmax(F.relu(similarity), dim=2)  
        norm_adj = adj  
      return norm_adj, adj