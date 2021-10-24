import torch


class AdvLoss(torch.nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, inputs):
        inputs = inputs.softmax(dim=1)
        B, C = inputs.shape

        value = 1. / C
        targets = torch.zeros_like(inputs)
        torch.fill_(targets, value)
        # loss = -torch.sum(targets * torch.log(inputs), dim=1)
        loss = torch.sum(torch.abs(targets - inputs), dim=1)

        return loss.mean()


# class AdvLoss(torch.nn.Module):
#     def __init__(self, eps=1e-5):
#         super().__init__()
#         self.eps = eps
#
#     def forward(self, inputs):
#         inputs = inputs.softmax(dim=1)
#         loss = - torch.log(inputs + self.eps).mean(dim=1)
#         return loss.mean()