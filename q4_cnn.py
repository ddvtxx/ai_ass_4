"""Homework 4, Question 4: reproducible CIFAR-10 CNN experiment.

Run: python q4_cnn.py --epochs 12 --threads 4
Optional download only: python q4_cnn.py --download-only
The final epoch is reported; test performance is never used to select a checkpoint.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import random
import sys
import time

ROOT = Path(__file__).resolve().parent
# Optional assignment-local torchvision installation; no global environment changes.
if (ROOT / '.deps').exists():
    sys.path.insert(0, str(ROOT / '.deps'))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision.datasets import CIFAR10
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = ROOT / 'output' / 'q4'
DATA = ROOT / 'data'

class SmallCNN(nn.Module):
    def __init__(self, second_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, second_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(second_channels * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        return self.fc2(self.relu(self.fc1(x)))

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_tensors(images, targets):
    # Same as ToTensor followed by Normalize((.5,)*3, (.5,)*3).
    x = torch.from_numpy(images.copy()).permute(0, 3, 1, 2).contiguous().float()
    x.div_(255.0).sub_(0.5).div_(0.5)
    return TensorDataset(x, torch.tensor(targets, dtype=torch.long))

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum = 0.0
    predicted, actual = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += nn.functional.cross_entropy(logits, y, reduction='sum').item()
        predicted.append(logits.argmax(1).cpu())
        actual.append(y.cpu())
    pred, truth = torch.cat(predicted), torch.cat(actual)
    return {'loss': loss_sum/len(truth), 'accuracy': (pred == truth).float().mean().item(),
            'correct': int((pred == truth).sum()), 'count': len(truth)}, pred.numpy(), truth.numpy()

def layer_details(model):
    rows = []
    x = torch.zeros(1, 3, 32, 32)
    for name in ['conv1', 'relu', 'pool', 'conv2', 'relu', 'pool', 'flatten', 'fc1', 'relu', 'fc2']:
        if name == 'flatten':
            x = torch.flatten(x,1)
            count = 0
        else:
            layer = getattr(model, name)
            x = layer(x)
            count = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        rows.append({'layer':name,'output_shape':list(x.shape),'parameters':count})
    assert sum(r['parameters'] for r in rows) == sum(p.numel() for p in model.parameters())
    assert rows[0]['output_shape'] == [1,16,32,32]
    assert rows[0]['parameters'] == (3*3*3+1)*16 == 448
    return rows

def examples_figure(images, pred, truth, classes, name):
    # Deterministic, transparent selection: first six correct and first six incorrect.
    good = np.flatnonzero(pred == truth)[:6]
    bad = np.flatnonzero(pred != truth)[:6]
    assert len(good) == len(bad) == 6
    indices = np.r_[good,bad]
    fig, axes = plt.subplots(3,4,figsize=(11,8))
    records=[]
    for ax,i in zip(axes.flat,indices):
        ok = bool(pred[i]==truth[i])
        ax.imshow(images[i], interpolation='nearest')
        ax.set_title(f'Test #{i} | {"Correct" if ok else "Incorrect"}\nTrue: {classes[truth[i]]}\nPred: {classes[pred[i]]}',
                     fontsize=10,color='#177245' if ok else '#ab3030')
        ax.axis('off')
        records.append({'test_index':int(i),'true_label':classes[truth[i]],
                        'predicted_label':classes[pred[i]],'correct':ok})
    fig.suptitle(f'{name}: six correct and six incorrect predictions',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(OUT/f'{name}_predictions.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    write_json(OUT/f'{name}_examples.json',records)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--epochs',type=int,default=12)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--download-only',action='store_true')
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads)
    print('Loading official CIFAR-10 dataset...',flush=True)
    train_raw=CIFAR10(root=str(DATA),train=True,download=True)
    test_raw=CIFAR10(root=str(DATA),train=False,download=True)
    assert len(train_raw)==50000 and len(test_raw)==10000
    if args.download_only:
        print('CIFAR-10 download and integrity checks complete.',flush=True)
        return
    torch.use_deterministic_algorithms(True)
    indices=np.random.default_rng(42).choice(len(train_raw),10000,replace=False)
    assert len(np.unique(indices))==10000
    np.save(OUT/'training_indices.npy',indices)
    labels=np.asarray(train_raw.targets)[indices]
    train=make_tensors(train_raw.data[indices],labels)
    test=make_tensors(test_raw.data,test_raw.targets)
    eval_train=DataLoader(train,batch_size=256,shuffle=False,num_workers=0)
    eval_test=DataLoader(test,batch_size=256,shuffle=False,num_workers=0)
    config={
        'seed':42,'shuffle_seed':2026,'subset_size':10000,'test_size':10000,
        'epochs':args.epochs,'batch_size':128,'optimizer':'Adam','learning_rate':.001,
        'weight_decay':.0001,'loss':'CrossEntropyLoss','augmentation':'none',
        'normalization':'ToTensor; channel mean=.5, channel std=.5',
        'device':args.device,'threads':args.threads,'checkpoint_selection':'fixed final epoch',
        'architecture_change':'conv2 output channels: 32 -> 64; fc1 input width follows automatically',
        'expectation':'More second-layer filters increase capacity and parameter count; training fit may improve, but test accuracy may or may not improve because only 10,000 training images are used.',
        'python':sys.version,'torch':str(torch.__version__),'torchvision':str(torchvision.__version__),
        'numpy':np.__version__,'matplotlib':matplotlib.__version__,
        'platform':platform.platform(),'subset_sha256':hashlib.sha256(indices.tobytes()).hexdigest(),
        'class_names':train_raw.classes,'training_class_counts':np.bincount(labels,minlength=10).tolist(),
        'sources':['https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html',
                   'https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html']
    }
    write_json(OUT/'experiment_config.json',config)
    all_history=[]
    summary=[]
    device=torch.device(args.device)
    for name,ch in [('baseline',32),('wider_conv2',64)]:
        seed_all(42)
        model=SmallCNN(ch)
        details=layer_details(model)
        write_json(OUT/f'{name}_architecture.json',details)
        model=model.to(device)
        loader=DataLoader(train,batch_size=128,shuffle=True,num_workers=0,
                          generator=torch.Generator().manual_seed(2026))
        optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
        criterion=nn.CrossEntropyLoss()
        total_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{name}: {total_params:,} trainable parameters',flush=True)
        start=time.perf_counter()
        for epoch in range(1,args.epochs+1):
            t=time.perf_counter()
            model.train()
            for x,y in loader:
                x,y=x.to(device),y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss=criterion(model(x),y)
                loss.backward()
                optimizer.step()
            train_metric,_,_=evaluate(model,eval_train,device)
            test_metric,pred,truth=evaluate(model,eval_test,device)
            row={'model':name,'epoch':epoch,'train_loss':train_metric['loss'],
                 'train_accuracy':train_metric['accuracy'],'test_loss':test_metric['loss'],
                 'test_accuracy':test_metric['accuracy'],'epoch_seconds':time.perf_counter()-t}
            all_history.append(row)
            with (OUT/'training_history.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(all_history)
            print(f'{name} epoch {epoch:02d}/{args.epochs}: train {row["train_accuracy"]:.2%}, test {row["test_accuracy"]:.2%}, {row["epoch_seconds"]:.1f}s',flush=True)
        torch.save({'state_dict':model.cpu().state_dict(),'second_channels':ch,'config':config,
                    'train_metrics':train_metric,'test_metrics':test_metric},OUT/f'{name}.pt')
        with (OUT/f'{name}_test_predictions.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['test_index','true_class','predicted_class','correct'])
            w.writerows((i,train_raw.classes[t],train_raw.classes[p],int(t==p)) for i,(t,p) in enumerate(zip(truth,pred)))
        examples_figure(test_raw.data,pred,truth,train_raw.classes,name)
        cm=np.zeros((10,10),dtype=int)
        np.add.at(cm,(truth,pred),1)
        np.savetxt(OUT/f'{name}_confusion_matrix.csv',cm,fmt='%d',delimiter=',',header=','.join(train_raw.classes),comments='')
        summary.append({'model':name,'conv2_channels':ch,'parameters':total_params,
                        'train':train_metric,'test':test_metric,'total_seconds':time.perf_counter()-start})
        write_json(OUT/'results.json',summary)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,color in [('baseline','#2563a8'),('wider_conv2','#bb5b28')]:
        h=[r for r in all_history if r['model']==name]
        ep=[r['epoch'] for r in h]
        for split,ls in [('train','-'),('test','--')]:
            axes[0].plot(ep,[r[f'{split}_accuracy']*100 for r in h],ls,color=color,label=f'{name} {split}')
            axes[1].plot(ep,[r[f'{split}_loss'] for r in h],ls,color=color,label=f'{name} {split}')
    axes[0].set_ylabel('Accuracy (%)');axes[1].set_ylabel('Cross-entropy loss')
    for ax in axes:
        ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Controlled experiment: second convolution 32 vs. 64 filters')
    fig.tight_layout();fig.savefig(OUT/'training_curves.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    main()
