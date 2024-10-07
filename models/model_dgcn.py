
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential as Seq
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GINConv
from models.model_utils import *

######################################
# DeepGraphConv Implementation #
######################################
class DeepGraphConv(torch.nn.Module):
    def __init__(self, edge_agg='latent', resample=0, num_features=1024, hidden_dim=256, 
        linear_dim=256, use_edges=False, dropout=0.25, n_classes=4):
        super(DeepGraphConv, self).__init__()
        self.use_edges = use_edges
        self.resample = resample
        self.edge_agg = edge_agg

        if self.resample > 0:
            self.fc = nn.Sequential(*[nn.Dropout(self.resample)])

        self.conv1 = GINConv(Seq(nn.Linear(num_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv2 = GINConv(Seq(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv3 = GINConv(Seq(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        
        self.path_attention_head = Attn_Net_Gated(L=hidden_dim, D=hidden_dim, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        self.classifier = torch.nn.Linear(hidden_dim, n_classes)

    def relocate(self):
        from torch_geometric.nn import DataParallel
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.conv1 = nn.DataParallel(self.conv1, device_ids=device_ids).to('cuda:0')
            self.conv2 = nn.DataParallel(self.conv2, device_ids=device_ids).to('cuda:0')
            self.conv3 = nn.DataParallel(self.conv3, device_ids=device_ids).to('cuda:0')
            self.path_attention_head = nn.DataParallel(self.path_attention_head, device_ids=device_ids).to('cuda:0')

        self.path_rho = self.path_rho.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, **kwargs):
        data = kwargs['x_path']
        x = data.x
        if self.edge_agg == 'spatial':
            edge_index = data.edge_index
        elif self.edge_agg == 'latent':
            edge_index = data.edge_latent
        batch = data.batch
        edge_attr = None

        if self.resample:
            x = self.fc(x)

        x1 = F.relu(self.conv1(x=x, edge_index=edge_index))
        x2 = F.relu(self.conv2(x1, edge_index, edge_attr))
        x3 = F.relu(self.conv3(x2, edge_index, edge_attr))
        

        h_path = x3

        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1) , h_path)
        h_path = self.path_rho(h_path).squeeze()
        h = h_path # [256] vector
        
        logits  = self.classifier(h).unsqueeze(0) # logits needs to be a [1 x 4] vector 
        Y_prob = F.softmax(logits, dim = 1)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        return logits, Y_prob, Y_hat, 0, 0 # The last two return are just dummy vars