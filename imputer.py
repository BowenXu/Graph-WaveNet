import torch
import torch.nn as nn
import torch.nn.functional as F


class Imputer(nn.Module):
    """Impute missing data based on type."""
    def __init__(self, impute_type, n_nodes, n_dim, seq_len=12,
                 gcn_dropout=0.0, gcn_support_len=1, gcn_order=1):
        super(Imputer, self).__init__()
        self.type = impute_type
        self.seq_len = seq_len
        self.n_nodes = n_nodes
        self.n_dim = n_dim

        if self.type == "ADJ":
            self.gcn = nconv()
        if self.type == "GCN":
            self.gcn = gcn(c_in=n_dim, c_out=n_dim, dropout=gcn_dropout,
                           support_len=gcn_support_len, order=gcn_order)


    def forward(self, x, supports=None):
        imputed_x = x
        if self.type != "":
            indices = (x == float("-inf")).nonzero(as_tuple=True)
            if self.type == "ZERO":
                imputed_x[indices] = 0.0
            elif self.type == "LAST":
                for _ in range(1, self.seq_len):
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
                    batch = index[0].item()
                    dim = index[1].item()
                    node = index[2].item()
                    seq = index[3].item()

                    try:
                        imputed_x[batch, dim, node, seq] = lookup[batch, node]
                    except:
                        mean = imputed_x[batch, dim, node, :][
                            imputed_x[batch, dim, node, :] !=
                            float("-inf")].mean()
                        mean = 0.0 if mean == float("-inf") or \
                                      torch.isnan(mean) else mean
                        imputed_x[batch, dim, node, seq] = mean
                        lookup[batch, node] = mean
            elif self.type in ["ADJ", "GCN"]:
                supports = supports[0] if self.type == "ADJ" else supports
                imputed_x[indices] = 0.0
                gcn_x = self.gcn(imputed_x, supports)
                imputed_x[indices] = gcn_x[indices]
            else:
                return NotImplementedError()


        return imputed_x.contiguous()


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()


    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))

        return x.contiguous()


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout=0.0, support_len=3, order=2):
        super(gcn, self).__init__()
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

        h = torch.cat(out, dim=1)
        h = self.mlp(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = F.dropout(h, self.dropout, training=self.training)

        return h


if __name__ == "__main__":
    SEQ_LENGTH = 3
    BATCH_SIZE = 1
    N_NODES = 2
    DIM = 1

    # [[[1, 2, 3]
    #   [4, 5, 6]]]
    data = torch.arange(1, 1 + N_NODES * SEQ_LENGTH, dtype=float).reshape(
        BATCH_SIZE, DIM, N_NODES, SEQ_LENGTH)

    # [[[-inf, 2, 3]
    #   [4, 5, -inf]]]
    left_missing = data.scatter(
        -1, torch.tensor([[[[0], [2]]]]), float("-inf"))

    # [[[-inf, -inf, 3]
    #   [4, -inf, -inf]]]
    block_missing = data.scatter(
        -1, torch.tensor([[[[0, 1], [1, 2]]]]), float("-inf"))

    # [[[-inf, -inf, -inf]
    #   [4, 5, 6]]]
    row_missing = data.scatter(
        -1, torch.tensor([[[[0, 1, 2], ]]]), float("-inf"))

    #
    # NON IMPUTATION
    #
    non_impute = Imputer("", N_NODES, DIM, SEQ_LENGTH)
    imputed_left_missing = non_impute(left_missing)
    imputed_data = non_impute(data)
    assert (left_missing == imputed_left_missing).all()
    assert (data == imputed_data).all()

    #
    # IMPUTATION WITH ZEROS
    #
    zero_impute = Imputer("ZERO", N_NODES, DIM, SEQ_LENGTH)

    # [[[-inf, 2, 3]     --->    [[[0.0, 2, 3]
    #   [4, 5, -inf]]]             [4, 5, 0.0]]]
    left_missing_ground_truth = data.scatter(
        -1, torch.tensor([[[[0], [2]]]]), 0.0)
    imputed_data = zero_impute(left_missing)
    assert (imputed_data == left_missing_ground_truth).all()

    # [[[-inf, -inf, 3]     --->    [[[0.0, 0.0, 3]
    #   [4, -inf, -inf]]]             [4, 0.0, 0.0]]]
    block_missing_ground_truth = data.scatter(
        -1, torch.tensor([[[[0, 1], [1, 2]]]]), 0.0)
    imputed_data = zero_impute(block_missing)
    assert (imputed_data == block_missing_ground_truth).all()

    # [[[-inf, -inf, -inf]     --->    [[[0.0, 0.0, 0.0]
    #   [4, 5, 6]]]                      [4, 5, 6]]]
    row_missing_ground_truth = data.scatter(
        -1, torch.tensor([[[[0, 1, 2], ]]]), 0.0)
    imputed_data = zero_impute(row_missing)
    assert (imputed_data == row_missing_ground_truth).all()

    #
    # IMPUTATION WITH LAST VALUE IN WINDOW, DEFAULT TO 0.0
    #
    last_impute = Imputer("LAST", N_NODES, DIM, SEQ_LENGTH)

    # [[[-inf, 2, 3]
    #   [4, 5, -inf]]]
    left_missing = data.scatter(
        -1, torch.tensor([[[[0], [2]]]]), float("-inf"))

    # [[[-inf, -inf, 3]
    #   [4, -inf, -inf]]]
    block_missing = data.scatter(
        -1, torch.tensor([[[[0, 1], [1, 2]]]]), float("-inf"))

    # [[[-inf, -inf, -inf]
    #   [4, 5, 6]]]
    row_missing = data.scatter(
        -1, torch.tensor([[[[0, 1, 2], ]]]), float("-inf"))

    # [[[-inf, 2, 3]     --->    [[[0.0, 2, 3]
    #   [4, 5, -inf]]]             [4, 5, 5]]]
    left_missing_ground_truth = data.scatter(
        -1, torch.tensor([[[[0], ]]]), 0.0)
    left_missing_ground_truth[0, 0, 1, 2] = 5
    imputed_data = last_impute(left_missing)
    assert (imputed_data == left_missing_ground_truth).all()

    # [[[-inf, -inf, 3]     --->    [[[0.0, 0.0, 3]
    #   [4, -inf, -inf]]]             [4, 4, 4]]]
    block_missing_ground_truth = data.scatter(
        -1, torch.tensor([[[[0, 1], ]]]), 0.0)
    block_missing_ground_truth[0, 0, 1, 1] = \
    block_missing_ground_truth[0, 0, 1, 2] = 4.
    imputed_data = last_impute(block_missing)
    assert (imputed_data == block_missing_ground_truth).all()

    # [[[-inf, -inf, -inf]     --->    [[[0.0, 0.0, 0.0]
    #   [4, 5, 6]]]                      [4, 5, 6]]]
    row_missing_ground_truth = data.scatter(
        -1, torch.tensor([[[[0, 1, 2], ]]]), 0.0)
    imputed_data = last_impute(row_missing)
    assert (imputed_data == row_missing_ground_truth).all()

    #
    # IMPUTATION WITH MEAN VALUE IN WINDOW, DEFAULT TO 0.0
    #
    mean_impute = Imputer("MEAN", N_NODES, DIM, SEQ_LENGTH)

    # [[[-inf, 2, 3]
    #   [4, 5, -inf]]]
    left_missing = data.scatter(
        -1, torch.tensor([[[[0], [2]]]]), float("-inf"))

    # [[[-inf, -inf, 3]
    #   [4, -inf, -inf]]]
    block_missing = data.scatter(
        -1, torch.tensor([[[[0, 1], [1, 2]]]]), float("-inf"))

    # [[[-inf, -inf, -inf]
    #   [4, 5, 6]]]
    row_missing = data.scatter(
        -1, torch.tensor([[[[0, 1, 2], ]]]), float("-inf"))

    # [[[-inf, 2, 3]     --->    [[[2.5, 2, 3]
    #   [4, 5, -inf]]]             [4, 5, 4.5]]]
    left_missing_ground_truth = data.scatter(
        -1, torch.tensor([[[[0], ]]]), 2.5)
    left_missing_ground_truth[0, 0, 1, 2] = 4.5
    imputed_data = mean_impute(left_missing)
    assert (imputed_data == left_missing_ground_truth).all()

    # [[[-inf, -inf, 3]     --->    [[[3, 3, 3]
    #   [4, -inf, -inf]]]             [4, 4, 4]]]
    block_missing_ground_truth = data.scatter(
        -1, torch.tensor([[[[0, 1], ]]]), 3)
    block_missing_ground_truth[0, 0, 1, 1] = \
    block_missing_ground_truth[0, 0, 1, 2] = 4.
    imputed_data = mean_impute(block_missing)
    assert (imputed_data == block_missing_ground_truth).all()

    # [[[-inf, -inf, -inf]     --->    [[[0.0, 0.0, 0.0]
    #   [4, 5, 6]]]                      [4, 5, 6]]]
    row_missing_ground_truth = data.scatter(
        -1, torch.tensor([[[[0, 1, 2], ]]]), 0.0)
    imputed_data = mean_impute(row_missing)
    assert (imputed_data == row_missing_ground_truth).all()

    #
    # IMPUTATION WITH 1 HOP NEIGHBOUR'S AVERAGE, DEFAULT TO 0.0
    #
    adj = torch.tensor([[0.1, 0.9],
                        [0.6, 0.4]]).double()
    one_hop_impute = Imputer("ADJ", N_NODES, DIM, SEQ_LENGTH)

    # [[[-inf, 2, 3]
    #   [4, 5, -inf]]]
    left_missing = data.scatter(
        -1, torch.tensor([[[[0], [2]]]]), float("-inf"))

    # [[[-inf, -inf, 3]
    #   [4, -inf, -inf]]]
    block_missing = data.scatter(
        -1, torch.tensor([[[[0, 1], [1, 2]]]]), float("-inf"))

    # [[[-inf, -inf, -inf]
    #   [4, 5, 6]]]
    row_missing = data.scatter(
        -1, torch.tensor([[[[0, 1, 2], ]]]), float("-inf"))

    # [[[-inf, 2, 3]     --->    [[[3.6, 2, 3]
    #   [4, 5, -inf]]]             [4, 5, 1.8]]]
    left_missing_ground_truth = data.scatter(
        -1, torch.tensor([[[[0], ]]]), 3.6)
    left_missing_ground_truth[0, 0, 1, 2] = 1.8
    imputed_data = one_hop_impute(left_missing, [adj.T])
    assert torch.allclose(imputed_data, left_missing_ground_truth)

    # [[[-inf, -inf, 3]     --->    [[[3.6, 0, 3]
    #   [4, -inf, -inf]]]             [4, 0, 1.8]]]
    block_missing_ground_truth = data.scatter(
        -1, torch.tensor([[[[0], ]]]), 3.6)
    block_missing_ground_truth[0, 0, 1, 2] = 1.8
    block_missing_ground_truth = block_missing_ground_truth.scatter(
        -1, torch.tensor([[[[1], [1]]]]), 0.0)
    imputed_data = one_hop_impute(block_missing, [adj.T])
    assert torch.allclose(imputed_data, block_missing_ground_truth)

    # [[[-inf, -inf, -inf]     --->    [[[3.6, 4.5, 5.4]
    #   [4, 5, 6]]]                      [4, 5, 6]]]
    row_missing_ground_truth = data.scatter(
        -1, torch.tensor([[[[0], ]]]), 3.6)
    row_missing_ground_truth = row_missing_ground_truth.scatter(
        -1, torch.tensor([[[[1], ]]]), 4.5)
    row_missing_ground_truth = row_missing_ground_truth.scatter(
        -1, torch.tensor([[[[2], ]]]), 5.4)
    imputed_data = one_hop_impute(row_missing, [adj.T])
    assert torch.allclose(imputed_data, row_missing_ground_truth)
