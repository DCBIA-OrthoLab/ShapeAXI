import torch
from torch import nn

class GaussianNoise(nn.Module):
    def __init__(self, mean=0.0, std=0.01):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if(self.training):
            return x + torch.normal(self.mean, self.std,size=x.shape, device=x.device)*(x!=0) # add noise on sphere (not on background)
        return x

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, input_seq):
        assert len(input_seq.size()) > 2

        size = input_seq.size()

        batch_size = size[0]
        time_steps = size[1]

        size_reshape = [batch_size*time_steps] + list(size[2:])
        reshaped_input = input_seq.contiguous().view(size_reshape)

        output = self.module(reshaped_input)

        output_size = output.size()
        output_size = [batch_size, time_steps] + list(output_size[1:])
        output = output.contiguous().view(output_size)

        return output

class MaxPoolImages(nn.Module):
    def __init__(self, nbr_images = 12):
        super().__init__()
        self.nbr_images = nbr_images
        self.max_pool = nn.MaxPool1d(self.nbr_images)

    def forward(self,x):
        x = x.permute(0,2,1)
        output = self.max_pool(x)
        output = output.squeeze(dim=2)

        return output

class AvgPoolImages(nn.Module):
    def __init__(self, nbr_images = 12):
        super().__init__()
        self.nbr_images = nbr_images
        self.avg_pool = nn.AvgPool1d(self.nbr_images)

    def forward(self,x):
        x = x.permute(0,2,1)
        output = self.avg_pool(x)
        output = output.squeeze(dim=2)

        return output

class SelfAttention(nn.Module):
    def __init__(self,in_units,out_units):
        super().__init__()


        self.W1 = nn.Linear(in_units, out_units)
        self.V = nn.Linear(out_units, 1)
        self.Tanh = nn.Tanh()
        self.Sigmoid = nn.Sigmoid()

    def forward(self, query, values):
        score = self.Sigmoid(self.V(self.Tanh(self.W1(query))))

        attention_weights = score/torch.sum(score, dim=1,keepdim=True)

        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector, score
