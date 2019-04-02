

#System
import numpy as np
import sys
import os
import random
from glob import glob
from skimage import io
from PIL import Image
import random
#Torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch
import torchvision.transforms as standard_transforms
from torchsummary import summary
#from models import LinkNet34
from linknet import LinkNet
from dataloader import palmFromText

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#torch.cuda.set_device(1)
ckpt_path = 'ckpt'
exp_name = 'INSTRU-LinkNet_Binary'
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
if not os.path.exists(os.path.join(ckpt_path, exp_name)):
    os.makedirs(os.path.join(ckpt_path, exp_name))
args = {
    'num_class': 1,
    'ignore_label': 255,
    'num_gpus': 1,
    'start_epoch': 1,
    'num_epoch': 200,
    'batch_size': 4,
    'lr': 0.01,
    'lr_decay': 0.9,
    'dice': 0,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'snapshot': '',
    'opt': 'adam',
}
IMG_MEAN = np.array((0.26870, 0.14467, 0.0679), dtype=np.float32)

def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, torch.nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def tversky(y_true, y_pred, smooth=1., alpha=0.5):
    y_true_pos = y_true.contiguous()
    y_pred_pos = y_pred.contiguous()
    true_pos = (y_true_pos * y_pred_pos).sum().sum()
    false_neg = (y_true_pos * (1-y_pred_pos)).sum().sum()
    false_pos = (1-y_true_pos)*(y_pred_pos).sum().sum()
    
    return ((true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)).mean()

def tl(y_true, y_pred):
    return 1-tversky(y_true,y_pred, alpha=0.6)

def ftl(y_true, y_pred, gamma=0.75):
    x = 1-tversky(y_true, y_pred, alpha=0.6)
    return x**gamma

class CrossEntropyLoss2d(torch.nn.Module):
	def __init__(self, weight=None, size_average=True):
		super(CrossEntropyLoss2d, self).__init__()
		self.nll_loss = torch.nn.NLLLoss(weight, size_average)

	def forward(self, inputs, targets):
		return self.nll_loss(F.log_softmax(inputs), targets)


if __name__ == '__main__':
    mean_std = ([0.154, 0.150, 0.019], [0.0118, 0.0218, 0.0022])
    imsize=1440
    num_epochs = 100
    batch_size = 4

    t = transforms.Compose([transforms.Resize((imsize,imsize)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0,0,0],
                                                            [1,1,1])])

    train_data = palmFromText("data/", "train.txt", transform=t)
    val_data = palmFromText("data/", "trainval.txt", transform=t)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    
    img_dir = 'data/train.txt'
    dataset = palmFromText(img_dir, t)
    train_loader = DataLoader(dataset=dataset, batch_size=args['batch_size'], shuffle=True, num_workers=2,drop_last=True)
    
    #model = LinkNet34(num_classes=args['num_class'], pretrained=True)
    
    model = LinkNet(n_classes=args['num_class'])
    gpu_ids = range(args['num_gpus'])
    model = torch.nn.parallel.DataParallel(model, device_ids=gpu_ids)
    model = model.cuda()

    if args['opt'] == 'sgd':
        opt = optim.SGD(group_weight(model),
                              # model.parameters(),
                              lr=0.002,
                              momentum=0.99, weight_decay=args['weight_decay'])
    elif args['opt'] == 'adam':
        opt = optim.Adam(group_weight(model),
                               # model.parameters(),
                               lr=args['lr'], weight_decay=args['weight_decay'])

    #criterion = CrossEntropyLoss2d(size_average=True).cuda()
    model.train()
    epoch_iters = dataset.__len__() / args['batch_size']
    max_epoch = 150
    resume_epoch = 0
    #args['snapshot'] = 'epoch_' + str(resume_epoch) + '.pth.tar'
    #model.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'])))


    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, verbose=True)
    early_stopping = utils.EarlyStopping(patience=10, verbose=True)

    print('='*30)
    print('Training')
    print('='*30)

    epoch_train_loss = []
    epoch_val_loss = []
    epoch_train_dsc = []
    epoch_val_dsc = []


    for epoch in range(num_epochs):
        train_losses = []
        train_dsc = []
        val_losses = []
        val_dsc = []

        steps = 0
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            model.train()
            opt.zero_grad()
            preds = model(images)
            loss = losses.ftl(preds, masks)

            loss.backward()
            opt.step()
            
            train_losses.append(loss.item())
            train_dsc.append(losses.dice_score(preds, masks).item())


        else:        
            val_loss = 0
            val_acc = 0
            model.eval()
            with torch.no_grad():
                for inputs, masks in val_loader:
                    inputs, masks = inputs.to(device), masks.to(device)
                    preds = model.forward(inputs)
                    loss = losses.ftl(preds, masks)

                    val_losses.append(loss.item())
                    val_dsc.append(losses.dice_score(preds,masks).item())
                    #scheduler.step(loss)

        print('[%d]/[%d] Train Loss:%.4f\t Train Acc:%.4f\t Val Loss:%.4f\t Val Acc: %.4f'
                % (epoch+1, num_epochs, 
                np.mean(train_losses),  np.mean(train_dsc),  
                np.mean(val_losses),  np.mean(val_dsc)))
        
        epoch_train_loss.append(np.mean(train_losses))
        epoch_val_loss.append(np.mean(val_losses))
        epoch_train_dsc.append(np.mean(train_dsc))
        epoch_val_dsc.append(np.mean(val_dsc))
        
        early_stopping(np.average(val_losses), model)
        
        if early_stopping.early_stop:
            print("Early stopping at epoch: ", epoch)
            break

        print('='*30)
        print('Average DSC score =', np.array(val_dsc).mean())


  
        #snapshot_name = 'epoch_' + str(epoch)
        #torch.save(model.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth.tar'))

