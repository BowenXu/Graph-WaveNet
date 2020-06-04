import torch.optim as optim
from model import *
from imputer import Imputer
import util
class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit, impute_type):
        self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        self.imputer = Imputer(impute_type, num_nodes, in_dim, device=device,
            gcn_dropout=dropout, gcn_support_len=self.model.supports_len, gcn_order=2)
        self.model.to(device)
        self.imputer.to(device)
        self.optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.imputer.parameters()),
            lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.imputer.train()
        self.optimizer.zero_grad()
        input_ = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input_)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        non_inf_indices = (real != float("-inf")).nonzero(as_tuple=True)
        real_ = real[non_inf_indices]
        predict_ = predict[non_inf_indices]

        loss = self.loss(predict_, real_, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) +
                list(self.imputer.parameters()), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict_, real_, 0.0).item()
        rmse = util.masked_rmse(predict_, real_, 0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        self.imputer.eval()
        input_ = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input_)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        non_inf_indices = (real != float("-inf")).nonzero(as_tuple=True)
        real_ = real[non_inf_indices]
        predict_ = predict[non_inf_indices]

        loss = self.loss(predict_, real_, 0.0)
        mape = util.masked_mape(predict_, real_, 0.0).item()
        rmse = util.masked_rmse(predict_, real_, 0.0).item()

        return loss.item(), mape, rmse
