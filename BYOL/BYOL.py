import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T

from tqdm import tqdm

from data import *
from model import *
# helper functions

def default(val, def_val):
    return def_val if val is None else val

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


# main class

class BYOL(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer = -2,
        projection_size = 128,
        projection_hidden_size = 512,
        augment_fn = None,
        augment_fn2 = None,
        moving_average_decay = 0.99,
        use_momentum = True
    ):
        super().__init__()
        self.net = net

        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer, use_simsiam_mlp=not use_momentum)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        device = get_module_device(net)
        # print(device)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size, image_size, device=device), torch.randn(2, 3, image_size, image_size, device=device))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(
        self,
        v_1,
        v_2,
        return_embedding = False,
        return_projection = True
    ):
        assert not (self.training and v_1.shape[0] == 1 and v_2.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(v_1)

        image_one, image_two = v_1, v_2

        online_proj_one, _ = self.online_encoder(image_one)
        online_proj_two, _ = self.online_encoder(image_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj_one, _ = target_encoder(image_one)
            target_proj_two, _ = target_encoder(image_two)
            target_proj_one.detach_()
            target_proj_two.detach_()

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BYOL')
    # parser.add_argument('--mode', type=str, help='sup or semi or base')
    parser.add_argument('--epochs', default=150, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--load_epoch', default=0, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--y',  type=str, help='Number of sweeps over the dataset to train')
    parser.add_argument('--s', type=str, help='Number of sweeps over the dataset to train')
    parser.add_argument('--name', type=str, help='Number of sweeps over the dataset to train')

    # args parse
    args = parser.parse_args()
    epochs = args.epochs
    # mode = 'UnfairBYOL'
    mode = f'BYOL_{args.name}_degree=3,4,5_0.0001_10000'
    # mode = f'BYOL_{args.y}_{args.s}_20000_0.111_degree=6,7,8_nocolorjitter_scale=0.8'
    
    # data prepare

    # train_data = CelebA_Pair(img_path=f"/home/zfd/CelebA_degree/two_degree_{args.y}_{args.s}_20000_0.111_0.0001_10000_different_degree_norm",
    #              attr_path=f"/home/zfd/CelebA_degree/csv_list/list_attr_celeba_{args.y}_{args.s}_20000_0.111.csv",
    #              metadata_path=f'/home/zfd/CelebA_degree/csv_list/metadata_{args.y}_{args.s}_20000_0.111.csv',
    #              target=args.y, sensitive=args.s)
    train_data = CelebA_Pair(img_path=f"./data/two_degree_{args.name}_0.0001_10000_different_degree_norm",
                attr_path=f"./data/list_attr_celeba_{args.name}.csv",
                metadata_path=f'./data/metadata_{args.name}.csv',
                target=args.y, sensitive=args.s)


    model = BYOLResNet()
    # model = nn.DataParallel(model)
    model.to("cuda")

    # learner = BYOL(
    #     model,
    #     image_size = 224,
    #     hidden_layer = 'encoder.avgpool'
    # ).cuda()

    learner = BYOL(
        model,
        image_size = 224,
        hidden_layer = 'encoder.avgpool'
    )
    # learner = nn.DataParallel(learner)
    learner.to("cuda")

    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

    feature_dim = 128
    batch_size = 32

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True,
                                drop_last=True)

    if not os.path.exists('{}-models'.format(mode)):
        os.mkdir('{}-models'.format(mode))

    for epoch in range(1, args.epochs):
        train_bar = tqdm(train_loader)
        for images_raw, images_pos in train_bar:
            images_raw, images_pos = images_raw.cuda(), images_pos.cuda()
            loss = learner(images_raw, images_pos)
            loss = torch.mean(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average() # update moving average of target encoder
            train_bar.set_description('Train Epoch: [{}] Loss: {:.4f}'.format(epoch, loss))

        save_name_pre = 'feature-dim={}_batchsize={}_epoch={}'.format(feature_dim, batch_size, epoch)

        if epoch == 1 or epoch % 5 == 0:
            torch.save(model.state_dict(), '{}-models/{}-model-{}.pth'.format(mode, mode, save_name_pre))
