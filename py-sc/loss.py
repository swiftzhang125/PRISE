import torch as torch
import torch.nn as nn
import numpy as np


def loss_fn(predictions, targets):
    criterion = nn.CrossEntropyLoss()
    return criterion(predictions, targets)

def one_hot_target(predictions, targets):
    device = targets.get_device()
    targets = targets.cpu().detach()
    one_hot = np.zeros((predictions.size(0), 10))
    one_hot[np.arange(targets.size(0)), targets] = 1
    return torch.from_numpy(one_hot).to(device)

def gaussian_noise(one_hot):
    return torch.normal(0.0, 1.0, size=one_hot.size(), device=one_hot.get_device())


class starConvexLoss(nn.Module):
    def __init__(self, lam, mu, rho, n_samples, aggregation):
        super(starConvexLoss, self).__init__()
        self.lam = lam
        self.mu = mu
        self.rho = rho
        self.n_samples = n_samples
        self.norm = nn.Softmax(dim=1)
        if aggregation == 'max':
            self.agg = 'max'
        else:
            self.agg = 'sum'
    
    def forward(self, predictions, targets):
        sc = None
        one_hot = one_hot_target(predictions, targets)
        #print(targets.size())
        wstar = loss_fn(predictions, targets)
        for _ in range(self.n_samples):

            noise = gaussian_noise(one_hot)
            one_hot_lamnoise = self.lam * noise + one_hot
            #print(one_hot_lamnoise.size())
            one_hot_lamnoise = self.norm(one_hot_lamnoise)
            #print(one_hot_lamnoise.size())
            one_hot_noise = noise + one_hot
            one_hot_noise = self.norm(one_hot_noise)

            #print(predictions.size())
            #print(one_hot_noise.size())

            w = loss_fn(predictions, one_hot_noise)
            wtidle = loss_fn(predictions, one_hot_lamnoise)

            noise2 = torch.sum(noise ** 2)
            
            # equ1 equ2 equ3
            sc1 = torch.clamp(wstar - wtidle, min=0)
            sc2 = torch.clamp(wstar - w + self.mu * noise2 / 2, min=0)
            sc3 = torch.clamp(wtidle - (1-self.lam)*wstar + self.lam * w + self.mu * self.lam * (1 - self.lam) * noise2 / 2, min=0)

            # max
            if self.agg == 'max':
                if sc is None:
                    sc = sc1 + sc2 + sc3
                else:
                    if sc < sc1 + sc2 + sc3:
                        sc = sc1 + sc2 + sc3
            else:
                if sc is None:
                    sc = sc1 + sc2 + sc3
                    print('go')
                else:
                    sc += sc1 + sc2 + sc3
        return wstar +  self.rho * sc


def criterion(convex, lam=None, mu=None, rho=None, n_samples=None, aggregation=None):
    #print(convex)
    if convex:
        return starConvexLoss(lam, mu, rho, n_samples, aggregation)
    else:
        return nn.CrossEntropyLoss()



if __name__ == '__main__':
    model = starConvexLoss(0.1, 0.1, 0.1, 4).to('cuda')
    x = torch.randn(2, 10).to('cuda')
    y = torch.randint(low=0, high=8, size=(2,)).to('cuda')
    print(x.size())
    print(y)
    print(model(x, y))