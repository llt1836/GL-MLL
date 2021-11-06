import torch
import torch.nn as nn
from torch.nn import init
from models.second_order_adj import second_order
from models.second_order_adj import gen_A0


class DenseGCNConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GCNConv`.
    """
    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        super(DenseGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.weight = nn.Parameter(torch.Tensor(self.in_channels, out_channels))
        init.xavier_uniform_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            self.bias.data = init.constant_(self.bias.data, 0.0)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, adj, mask=None, add_loop=True):
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}`, with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature
                dimension :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
                \times N \times N}`. The adjacency tensor is broadcastable in
                the batch dimension, resulting in a shared adjacency matrix for
                the complete batch.
            mask (BoolTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not self.improved else 2

        out = torch.matmul(x, self.weight)
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.matmul(adj, out)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out


class CorrGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, num_nodes):  # num_nodes = num_classes
        super(CorrGraphConvolution, self).__init__()

        self.num_nodes = num_nodes
        self.relu = nn.LeakyReLU(0.2)

        self.fc = nn.Linear(in_features, out_features)

        self.dynamic_weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.kaiming_normal_(self.dynamic_weight)

        self.dynamic_weight2 = nn.Parameter(torch.FloatTensor(out_features, out_features))
        nn.init.kaiming_normal_(self.dynamic_weight2)

        _adj = gen_A0('get_gcn_x_adj/train_adj_14.pkl')

        _adj1, _adj2 = second_order(_adj)
        result = torch.FloatTensor(_adj).to('cuda')

        result1 = torch.FloatTensor(_adj1).to('cuda')
        result2 = torch.FloatTensor(_adj2).to('cuda')
        self.corr = result

        self.corr1 = result1
        self.corr2 = result2

        self.gcnn = DenseGCNConv(in_channels=in_features, out_channels=int(out_features/2))
        self.gcnn2 = DenseGCNConv(in_channels=int(out_features/2), out_channels=out_features)
        self.norm = nn.BatchNorm1d(num_features=14)
        self.norm2 = nn.BatchNorm1d(num_features=14)
        self.norm3 = nn.BatchNorm1d(num_features=14)

    def calculate_correlation_matrix(self, x):
        correlation = torch.ones((x.shape[0], self.num_nodes, self.num_nodes)).to('cuda')
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                correlation[:, i, j] = torch.abs(torch.cosine_similarity(x[:, i, :], x[:, j, :], dim=-1))
                correlation[:, j, i] = correlation[:, i, j]
        return correlation

    def forward(self, x):

        h = self.fc(x)
        h = self.relu(h)

        corr = self.calculate_correlation_matrix(h)

        out = self.gcnn(x, self.corr1)
        out = self.norm(out)
        out = self.relu(out)

        out3 = self.gcnn(x, self.corr2)
        out3 = self.norm3(out3)
        out3 = self.relu(out3)

        out4 = self.gcnn2(out3, corr)
        out4 = self.norm2(out4)
        out4 = self.relu(out4)

        out2 = self.gcnn2(out, corr)
        out2 = self.norm2(out2)
        out2 = self.relu(out2)
        output = out2+out4

        return output, corr, h


class MyGCN(nn.Module):
    def __init__(self, model, num_classes):
        super(MyGCN, self).__init__()

        self.num_classes = num_classes

        # resnet:
        self.features1 = nn.Sequential(
            model.conv1,
            model.layer1,
            model.layer2,
            model.layer3,
        )

        self.features2 = nn.Sequential(
            model.layer4,
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        # resnet
        self.fc = model.fc

        self.relu = nn.LeakyReLU(0.2)

        # # resnet
        self.gcn = CorrGraphConvolution(49, 2048, num_classes)

        self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())
        self.last_linear = nn.Conv1d(512, self.num_classes, 1)

        self.last_fc = nn.Linear(1024, 14)
        self.sig = nn.Sigmoid()

        self.conv2 = nn.Conv2d(2048, 14, 1)
        self.z_norm = nn.BatchNorm1d(14)

    def forward_feature(self, x):
        x = self.features1(x).detach()
        x = self.features2(x)
        return x

    def forward_sam(self, x):
        x1 = self.gap(x).squeeze()
        x1 = self.fc(x1)
        out1 = self.sig(x1)
        x = torch.matmul(x.transpose(3, 1), self.fc.weight.transpose(1, 0))
        x = x.transpose(3, 1)
        x = x.reshape(x.size(0), x.size(1), -1)
        x = self.relu(x)
        return x, out1

    def forward_dgcn(self, x):
        x, corr, h = self.gcn(x)
        return x, corr, h

    def classifier(self, x, z):
        x = self.gap(x)
        if x.shape[0] == 1:
            x = x.squeeze().unsqueeze(0).unsqueeze(1)
        else:
            x = x.squeeze().unsqueeze(1)

        out2 = torch.matmul(x, z.transpose(2, 1))
        if out2.shape[0] == 1:
            out2 = torch.matmul(x, z.transpose(2, 1)).squeeze(1)
        else:
            out2 = torch.matmul(x, z.transpose(2, 1)).squeeze()

        out2 = self.z_norm(out2)
        out2 = self.sig(out2)
        output = out2

        return output

    def forward(self, x):
        x = self.forward_feature(x)

        v, out1 = self.forward_sam(x)
        z, corr, h = self.forward_dgcn(v)

        out2 = self.classifier(x, z)

        return out1, out2, corr, x

    def classifier_loss(self, input1, input2, target, corr):
        criterion = MyLoss().to('cuda')
        loss0 = criterion(input1, target)
        loss1 = criterion(input2, target)
        loss = loss1 + loss0

        return loss

    def dice_loss_input(self, corr, target):
        target = target.unsqueeze(1).transpose(2, 1)
        gt = torch.matmul(target, target.transpose(2, 1))
        loss = self.dice_loss(corr, gt)
        return loss

    def dice_loss(self, input, target):
        N = target.size(0)
        smooth = 1
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
        intersection = input_flat * target_flat
        loss = (2 * intersection.sum(1)) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N
        return loss


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def correlation_loss(self, corr, target):
        target = target.unsqueeze(-1)  # 32*14*1
        loss = torch.matmul(corr, target).squeeze()  # 32, 14
        loss = loss.reshape(-1)
        return loss

    def forward(self, input, target):
        input = input.reshape(-1)
        target = target.reshape(-1)
        wp, wn = self.calculate_weight(target)
        loss1 = 0
        loss0 = 0

        for i in range(len(target)):
            if target[i] == 1:
                tmp = -(wp+1)*torch.log(input[i])
                loss1 += tmp
            elif target[i] == 0:
                tmp = -(wn+1)*torch.log(1-input[i])
                loss0 += tmp
        loss = (loss1 + loss0)/len(target)
        return loss

    def calculate_weight(self, target):
        length = len(target)+1
        wp = length/(torch.sum(target)+1)
        wn = length/(length-torch.sum(target))
        return wp, wn
