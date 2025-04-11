# -*- coding: utf-8 -*-
# Torch
import torch.nn as nn
import torch
import torch.optim as optim

# utils
import os
import datetime
import numpy as np
import joblib
from torch.nn import functional as F
from tqdm import tqdm
from utils import grouper, sliding_window, count_sliding_window, camel_to_snake
from model.BPFTNet import BPFTNet
from losses import Cross_fusion_CNN_Loss, EndNet_Loss
eps = 1e-7


class SCELoss(nn.Module):
    def __init__(self, num_classes=10, a=1, b=1):
        super(SCELoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.b = b
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        ce = self.cross_entropy(pred, labels)
        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        loss = self.a * ce + self.b * rce.mean()
        return loss
    
class AExpLoss(torch.nn.Module):
    def __init__(self, num_classes=10, a=8, scale=1.0,e=0.6):
        super(AExpLoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.scale = scale
        self.logsoftmax=nn.LogSoftmax(dim=1)
        self.e = e

    def forward(self, pred, labels):
        pred = self.logsoftmax(pred)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = torch.exp(-torch.sum(label_one_hot * pred, dim=1) / self.a) *(1-self.e) + self.e/self.num_classes
        return loss.mean() * self.scale

class CrossEntropyLabelSmoothLoss(nn.Module):
    def __init__(self,num_classes=767,e=0.6,use_gpu=True):
        super(CrossEntropyLabelSmoothLoss,self).__init__()
        self.num_classes=num_classes    # 类别个数767
        self.e=e    # 超参数e
        self.use_gpu=use_gpu    # 是否使用GPU
        self.logsoftmax=nn.LogSoftmax(dim=1)    # LogSoftmax函数
    def forward(self,inputs,targets):
        '''
        输入:
        inputs: cls_feat特征向量,type为Tensor,shape为8*767;
        targets: 1个batch中8张图像的行人索引值,取值为0~766,type为Tensor,是长度为8的一维向量
        输出:
        交叉熵损失值,type为Tensor,一个数值;
        '''
        log_probs=self.logsoftmax(inputs)   # 对cls_feat做LogSoftmax变换
        targets=torch.zeros(log_probs.size()).scatter_(1,targets.unsqueeze(1).cpu(),1)  # 索引值生成标签值
        if self.use_gpu:    # 是否使用cuda加速
            targets=targets.cuda()
        targets=(1-self.e)*targets+self.e/self.num_classes  # Label Smooth
        loss=(-targets*log_probs).mean(0).sum()     # 基于标签值和预测值计算交叉熵损失
        return loss

    

def get_model(name,bn_threshold, **kwargs):
    """
    Instantiate and obtain a model with adequate hyperparameters

    Args:
        name: string of the model name
        kwargs: hyperparameters
    Returns:
        model: PyTorch network
        optimizer: PyTorch optimizer
        criterion: PyTorch loss Function
        kwargs: hyperparameters with sane defaults
    """
    device = kwargs.setdefault("device", torch.device("cuda"))
    n_classes = kwargs["n_classes"]
    (n_bands, n_bands2) = kwargs["n_bands"]
    weights = torch.ones(n_classes)
    weights[torch.LongTensor(kwargs["ignored_labels"])] = 0.0
    weights = weights.to(device)
    weights = kwargs.setdefault("weights", weights)

 
    if name == "BPFTNet":
        kwargs.setdefault("patch_size", 9)
        center_pixel = True
        model = BPFTNet(n_bands, n_bands2, n_classes, kwargs["patch_size"],bn_threshold)
        lr = kwargs.setdefault("lr", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = AExpLoss(num_classes=n_classes)
        kwargs.setdefault("epoch", 600)
        kwargs.setdefault("batch_size", 64)
    else:
        raise KeyError("{} model is unknown.".format(name))

    model = model.to(device)
    epoch = kwargs.setdefault("epoch", 100)
    kwargs.setdefault(
        "scheduler",
        # optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, factor=0.1, patience=epoch // 4, verbose=True
        # ),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch),
        # torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 150, 180], gamma=0.1),
    )
    # kwargs.setdefault('scheduler', None)
    kwargs.setdefault("batch_size", 64)
    kwargs.setdefault("supervision", "full")
    kwargs.setdefault("flip_augmentation", False)
    kwargs.setdefault("radiation_augmentation", False)
    kwargs.setdefault("mixture_augmentation", False)
    kwargs["center_pixel"] = center_pixel
    return model, optimizer, criterion, kwargs



def train(
    net,
    optimizer,
    criterion,
    data_loader,
    epoch,
    scheduler=None,
    display_iter=100,
    device=torch.device("cuda"),
    display=None,
    val_loader=None,
    supervision="full"
):
    """
    Training loop to optimize a network for several epochs and a specified loss

    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        data_loader: a PyTorch dataset loader
        epoch: int specifying the number of training epochs
        criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
        device (optional): torch device to use (defaults to CPU)
        display_iter (optional): number of iterations before refreshing the
        display (False/None to switch off).
        scheduler (optional): PyTorch scheduler
        val_loader (optional): validation dataset
        supervision (optional): 'full' or 'semi'
    """

    n_classes = 21
    weights = torch.ones(n_classes)
    weights[torch.LongTensor([0])] = 0.0
    weights = weights.to(device)
    criterxy = nn.CrossEntropyLoss(weight=weights)
    
    if criterion is None:
        raise Exception("Missing criterion. You must specify a loss function.")

    net.to(device)
    save_epoch = epoch // 20 if epoch > 20 else 1

    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    iter_ = 1
    loss_win, val_win = None, None
    val_accuracies = []
    Best_accuracy = 0
    for e in tqdm(range(1, epoch + 1), desc="Training the network"):
        # Set the network to training mode
        net.train()
        avg_loss = 0.0

        

        
        # Run the training loop for one epoch
        for batch_idx, (data, data2,data_pca, target) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            # Load the data into the GPU if required
            data, data2, target = data.to(device), data2.to(device), target.to(device)
            data_pca = data_pca.to(device)
            
            optimizer.zero_grad()
            if supervision == "full":
                lx,ly,output = net(data, data2)
                # output = net(data, data2)
                a = 0.2
                b = 0.9
                c = 0.9
                # wloss = criterion(fus,target)
                lossx = criterion(lx,target)
                lossy = criterion(ly,target)
                loss = criterion(output, target) * a + lossx * b  + lossy * c
                
                
            elif supervision == "semi":
                outs = net(data, data2)
                output, rec = outs
                loss = criterion[0](output, target) + net.aux_loss_weight * criterion[
                    1
                ](rec, data)
            else:
                raise ValueError(
                    'supervision mode "{}" is unknown.'.format(supervision)
                )
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100) : iter_ + 1])

            if display_iter and iter_ % display_iter == 0:
                string = "Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.9f}"
                string = string.format(
                    e,
                    epoch,
                    batch_idx * len(data),
                    len(data) * len(data_loader),
                    100.0 * batch_idx / len(data_loader),
                    mean_losses[iter_],
                )
                update = None if loss_win is None else "append"
                tqdm.write(string)

                
            iter_ += 1
            del (data, target, loss, output)

        # Update the scheduler
        avg_loss /= len(data_loader)
        if val_loader is not None:
            val_acc = val(net, val_loader, device=device, supervision=supervision)
            if val_acc > Best_accuracy:
                torch.save(net.state_dict(), "/tmp/output/best_val_model.pth")
                print("the epoch {} acc:{} ".format(e,val_acc))
            val_accuracies.append(val_acc)
            metric = -val_acc
        else:
            metric = avg_loss

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        elif scheduler is not None:
            scheduler.step()

        # Save the weights
        if e % save_epoch == 0:
            save_model(
                net,
                camel_to_snake(str(net.__class__.__name__)),
                data_loader.dataset.name,
                epoch=e,
                metric=abs(metric),
            )


def save_model(model, model_name, dataset_name, **kwargs):
    model_dir = "./checkpoints/" + model_name + "/" + dataset_name + "/"
    """
    Using strftime in case it triggers exceptions on windows 10 system
    """
    time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if isinstance(model, torch.nn.Module):
        filename = time_str + "_epoch{epoch}_{metric:.2f}".format(
            **kwargs
        )
        tqdm.write("Saving neural network weights in {}".format(filename))
        torch.save(model.state_dict(), model_dir + filename + ".pth")
    else:
        filename = time_str
        tqdm.write("Saving model params in {}".format(filename))
        joblib.dump(model, model_dir + filename + ".pkl")


def test(net, img1, img2, hyperparams):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size = hyperparams["patch_size"]
    center_pixel = hyperparams["center_pixel"]
    batch_size, device = hyperparams["batch_size"], hyperparams["device"]
    n_classes = hyperparams["n_classes"]

    kwargs = {
        "step": hyperparams["test_stride"],
        "window_size": (patch_size, patch_size),
    }
    probs = np.zeros(img1.shape[:2] + (n_classes,))

    iterations = count_sliding_window(img1, img2, **kwargs) // batch_size
    for batch in tqdm(
        grouper(batch_size, sliding_window(img1, img2, **kwargs)),
        total=(iterations),
        desc="Inference on the image",
    ):
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)

                data2 = [b[1][0, 0] for b in batch]
                data2 = np.copy(data2)
                data2 = torch.from_numpy(data2)
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)
                # data = data.unsqueeze(1)

                data2 = [b[1] for b in batch]
                data2 = np.copy(data2)
                data2 = data2.transpose(0, 3, 1, 2)
                data2 = torch.from_numpy(data2)
                # data2 = data2.unsqueeze(1)
                
                data_pca = [b[2] for b in batch]
                data_pca = np.copy(data_pca)
                data_pca = data_pca.transpose(0, 3, 1, 2)
                data_pca = torch.from_numpy(data_pca)
                
            indices = [b[3:] for b in batch]
            data = data.to(device)
            data2 = data2.to(device)
            data_pca = data_pca.to(device)
            _,_,output = net(data, data2)
            # output = net(data,data2)
            if isinstance(output, tuple):  # For multiple outputs
                output = output[0]
            output = output.to("cpu")

            if patch_size == 1 or center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x + w // 2, y + h // 2] += out
                else:
                    probs[x : x + w, y : y + h] += out
    return probs



def val(net, data_loader, device="cpu", supervision="full"):
    # TODO : fix me using metrics()
    accuracy, total = 0.0, 0.0
    ignored_labels = data_loader.dataset.ignored_labels
    for batch_idx, (data, data2,data_pca, target) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, data2, target = data.to(device), data2.to(device), target.to(device)
            data_pca = data_pca.to(device)
            if supervision == "full":
                _,_,output = net(data ,data2)
                # output = net(data ,data2)
            elif supervision == "semi":
                outs = net(data)
                output, rec = outs

            if isinstance(output, tuple):   # For multiple outputs
                output = output[0]
            _, output = torch.max(output, dim=1)
            for out, pred in zip(output.view(-1), target.view(-1)):
                if out.item() in ignored_labels:
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    total += 1
    return accuracy / total
