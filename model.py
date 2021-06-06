import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x


class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        rand_indices = torch.randperm(num_nodes).to(device)
        losses = []

        #for i in range(num_batches):
        #    mask = indices[i * batch_size:(i + 1) * batch_size]
        #    refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
        #    between_sim = f(self.sim(z1[mask], z2))  # [B, N]

        #    losses.append(-torch.log(
        #        between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
        #        / (refl_sim.sum(1) + between_sim.sum(1)
        #           - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        for i in range(num_batches):
            ordered_mask = indices[i * batch_size:(i + 1) * batch_size]
            random_mask = rand_indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[ordered_mask], z1[random_mask]))  # [B, N]
            between_sim = f(self.sim(z1[ordered_mask], z2[random_mask]))  # [B, N]

            #losses.append(-torch.log(
            #    f((F.normalize(z1[ordered_mask])*F.normalize(z2[ordered_mask])).sum(1))
            #    / (refl_sim.sum(1) + between_sim.sum(1))))
            losses.append(torch.log(refl_sim.sum(1) + between_sim.sum(1)) - (F.normalize(z1[ordered_mask])*F.normalize(z2[ordered_mask])).sum(1)/self.tau)

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


class SelfExpr(Module):
    def __init__(self, n):
        self.n = n
        super(SelfExpr, self).__init__()
        #self.weight = Parameter(1*torch.FloatTensor(n, n))
        self.weight = Parameter(torch.FloatTensor(n, n).uniform_(0,0.01))

    def forward(self, input):
        #self.weight.data = F.relu(self.weight)
        output = torch.mm(self.weight-torch.diag(torch.diagonal(self.weight)), input)
        return self.weight, output
    
    def reset(self, input):
        self.weight.data = torch.FloatTensor(self.n, self.n).uniform_(0,0.01)


class ClusterModel(torch.nn.Module):
    def __init__(self, n_hid1, n_hid2, n_class, dropout):
        super(ClusterModel, self).__init__()
        self.mlp1 = torch.nn.Linear(n_hid1, n_hid2)
        self.mlp2 = torch.nn.Linear(n_hid2, n_class)
        self.dropout = dropout
        # ~ torch.nn.init.xavier_uniform_(self.mlp1.weight)
        # ~ torch.nn.init.xavier_uniform_(self.mlp2.weight)
    
    def forward(self, x1: torch.Tensor) -> torch.Tensor:
            x2 = F.relu(self.mlp1(x1))
            if self.dropout > 0:
                x2 = F.dropout(x2, self.dropout, training=self.training)
            z = F.softmax(self.mlp2(x2), dim=-1)
            #z = F.relu(self.fc4(x3))
            return z
