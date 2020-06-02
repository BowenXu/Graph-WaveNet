import torch
import torch.nn as nn
import torch.nn.functional as F


class Imputer(nn.Module):
    """Impute missing data based on type."""
    def __init__(self, impute_type, n_nodes, n_dim, seq_len=12,
                 gcn_dropout=0.0, gcn_support_len=1, gcn_order=1):
        super(Imputer,self).__init__()
        self.type = impute_type
        self.seq_len = seq_len
        self.n_nodes = n_nodes
        self.n_dim = n_dim

        if self.type == "GCN":
            self.gcn = nconv()
            #self.gcn = gcn(c_in=n_dim, c_out=n_dim, dropout=gcn_dropout,
            #               support_len=gcn_support_len, order=gcn_order)


    def forward(self, x, supports=None):
        imputed_x = x
        if self.type != "":
            indices = (x == float("-inf")).nonzero(as_tuple=True)
            if self.type == "ZERO":
                imputed_x[indices] = 0.0
            elif self.type == "LAST":
                for i in range(1, self.seq_len):
                    seq_len_indices = indices[-1]
                    left_shifted_indices = \
                        torch.clamp(seq_len_indices - 1, min=0)
                    imputed_x[indices] = imputed_x[
                        (indices[0], indices[1], indices[2],
                         left_shifted_indices)]
                    indices = \
                        (imputed_x == float("-inf")).nonzero(as_tuple=True)
                    if len(indices[-1]) == 0 or \
                       (indices[-1] == 0).all() or \
                       (len(indices[-1]) == len(seq_len_indices)) and \
                       (indices[-1] == seq_len_indices).all():
                        break
                # default left-overs to zeros
                imputed_x[indices] = 0.0
            elif self.type == "MEAN":
                lookup = {}
                indices = torch.stack(indices, dim=1)
                for index in indices:
                    batch = index[0]
                    dim = index[1]
                    node = index[2]
                    seq = index[3]

                    try:
                        imputed_x[batch, dim, node, seq] = lookup[batch, node]
                    except:
                        mean = imputed_x[batch, dim, node, :][
                            imputed_x[batch, dim, node, :] !=
                            float("-inf")].mean()
                        mean = 0.0 if mean == float("-inf") else mean
                        imputed_x[batch, dim, node, seq] = mean
                        lookup[batch, node] = mean
            elif self.type == "GCN":
                imputed_x[indices] = 0.0
                gcn_x = self.gcn(imputed_x, supports)
                imputed_x[indices] = gcn_x[indices]
            else:
                return NotImplementedError()


        return imputed_x.contiguous()


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()


    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))

        return x.contiguous()


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout=0.0, support_len=3, order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = nn.Linear(c_in, c_out)
        self.dropout = dropout
        self.order = order


    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = F.dropout(h, self.dropout, training=self.training)

        return h
