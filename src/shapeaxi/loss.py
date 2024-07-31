class weighted_mse_loss():
    def __init__(self,labels,weight):

        self.labels = torch.tensor(labels)
        self.weight = torch.tensor(weight)

    def get_weights(self,target):
        device = target.device.type
        labels = self.labels.to(device,non_blocking=True)
        weight = self.weight.to(device,non_blocking=True)
        w = torch.zeros(target.shape).to(device,non_blocking=True)
        for i in range(len(self.labels)):
            w += (weight[i].item() * (target == labels[i].item()))
        return w

    def __call__(self,input,target):
        w = self.get_weights(target)
        loss = torch.sum(w*(input - target) ** 2)
        return loss

class weighted_mae_loss():
    def __init__(self,labels,weight):

        self.labels = torch.tensor(labels)
        self.weight = torch.tensor(weight)

    def get_weights(self,target):
        device = target.device.type
        labels = self.labels.to(device,non_blocking=True)
        weight = self.weight.to(device,non_blocking=True)
        w = torch.zeros(target.shape).to(device,non_blocking=True)
        for i in range(len(self.labels)):
            w += (weight[i].item() * (target == labels[i].item()))
        return w

    def __call__(self,input,target):
        w = self.get_weights(target)
        loss = torch.sum(w*torch.abs(input - target))
        return loss