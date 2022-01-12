import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import dataloader as dset
import numpy as np
import os
from msrfc import *
from msrfc import VGGnet_spatial # Importing SA-Net
from patchnet import * # Importing PatchNet
from pthflops import count_ops
class LabelSomCE(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self,x,target,smoothing=0.1):
		confidence = 1.0 - smoothing
		logprobs = F.log_softmax(x,dim=-1)
		nll_loss = - logprobs.gather(dim=-1,index=target.unsqueeze(1))
		nll_loss = nll_loss.squeeze(1)
		smooth_loss = -logprobs.mean(dim=-1)
		loss = confidence * nll_loss + smoothing * smooth_loss

		return loss.mean()

def download(folder,thetarfile):
    import urllib.request
    import tarfile
    ftpstream = urllib.request.urlopen(thetarfile)
    thetarfile = tarfile.open(fileobj=ftpstream, mode="r|gz")
    thetarfile.extractall(folder)
    thetarfile.close()
    
def download_cerug(folder):
    thetarfile = "https://www.ai.rug.nl/~sheng/writerset/CERUG-EN-train-images.tar.gz"
    download(folder,thetarfile)
    thetarfile = "https://www.ai.rug.nl/~sheng/writerset/CERUG-EN-test-images.tar.gz"
    download(folder,thetarfile)

def download_firemaker(folder):
    thetarfile = "https://www.ai.rug.nl/~sheng/writerset/Firemaker-train-images.tar.gz"
    download(folder,thetarfile)
    thetarfile = "https://www.ai.rug.nl/~sheng/writerset/Firemaker-test-images.tar.gz"
    download(folder,thetarfile)
    
class DeepWriter_Train:
    def __init__(self,dataset='Firemaker',imgtype='png',mode='vertical'):
    
        self.dataset = dataset
        self.folder = dataset
        #self.labelfolder = 'dataset/'
        if self.dataset == 'IAM':
            self.folder = 'data'
        if not os.path.exists(self.folder):
            if dataset == 'CERUG-EN':
                download_cerug(dataset)
            elif dataset == 'Firemaker':
                download_firemaker(dataset)
            else:
                print('****** Warning: the dataset %s does not existed!******'%dataset)
                print('Please go to the following website to check how to download the dataset:')
                print('https://www.ai.rug.nl/~sheng/writeridataset.html')
                print('*'*20)
                raise ValueError('Dataset: %s does not existed!'%dataset)
        
        self.labelfolder = self.folder
        self.train_folder = self.folder+'/train/'
        self.test_folder = self.folder+'/test/'
        
        self.imgtype=imgtype
        self.mode = mode
        self.device = 'cuda' #make sure to change this
        self.scale_size=(64,128)
        
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
        
        if self.dataset == 'CVL':
            self.imgtype = 'tif'
        
        self.model_dir = 'model'
        if not os.path.exists(self.model_dir):
            #raise ValueError('Model directory: %s does not existed'%self.model_dir)
            os.mkdir(self.model_dir)#raise ValueError('Model directory: %s does not existed'%self.model_dir)
        
        basedir = 'MSRF_firemaker_'+self.dataset+'_model_'+self.mode+'_aug_16'
        self.logfile= basedir + '.log'
        self.modelfile = basedir
        self.batch_size = 16
        
        train_set = dset.DatasetFromFolder(dataset=self.dataset,
        				labelfolder = self.labelfolder,
                        foldername=self.train_folder,
                        imgtype=self.imgtype,
                        scale_size=self.scale_size,
                        is_training = True)
        
        self.training_data_loader = DataLoader(dataset=train_set, num_workers=0, 
                           batch_size=self.batch_size, shuffle=True)
        
        test_set = dset.DatasetFromFolder(dataset=self.dataset,
        				labelfolder = self.labelfolder,
                        foldername=self.test_folder,imgtype=self.imgtype,
                        scale_size=self.scale_size,
                        is_training = False)
        
        self.testing_data_loader = DataLoader(dataset=test_set, num_workers=0, 
                           batch_size=self.batch_size, shuffle=False)
        
        num_class = train_set.num_writer
        #self.model = dfrag(1,num_classes=train_set.num_writer).to(self.device) Use this for PatchNet
        #self.model = VGGnet_spatial(1,train_set.num_writer).to(self.device) Use this for SA-Net
        self.model = MSF(1,train_set.num_writer).to(self.device)
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        print('Number of parameters is ',pytorch_total_params)
        self.criterion = nn.CrossEntropyLoss()
        self.criterion = LabelSomCE()
        self.optimizer = optim.Adam(self.model.parameters(),lr=0.0001,weight_decay=1e-4) 
        self.scheduler = lr_scheduler.StepLR(self.optimizer,step_size=10,gamma=0.5)
        self.page = np.zeros((train_set.num_writer,train_set.num_writer))
        #print('CALCULATING FLOPS')
        #inp  = torch.randn(1, 1, 64,128)
        #print(count_ops(self.model, inp))
        #print('#'*10,'DONE','#'*10)
        #self.test(48,during_train=True)           
    def train(self,epoch):
        self.model.train()
        losstotal = []
        
        for iteration,batch in enumerate(self.training_data_loader,1):
            inputs = batch[0].to(self.device).float()
            target = batch[1].type(torch.long).to(self.device)
            
            #self.optimizer.zero_grad() Uncomment this piece of code when using the SA-Net
            #train_loss = 0
            #logs = self.model(inputs)
            #train_loss= train_loss + self.criterion(logs,target)
        
            self.optimizer.zero_grad()
            train_loss = 0
            logits_list,combined_logits = self.model(inputs)
            for logs in logits_list:
                train_loss= train_loss + self.criterion(logs,target)

            losstotal.append(train_loss.item())
            train_loss.backward()
            self.optimizer.step()
        
        
        with open(self.logfile,'a') as fp:
            fp.write('Training epoch %d avg loss is: %.6f\n'%(epoch,np.mean(losstotal)))
        print('Traing epoch:',epoch,'  avg loss is:',np.mean(losstotal))
    def test(self,epoch,during_train=True):
        self.model.eval()
        
        if not during_train:
            self.load_model(epoch)

        top1 = 0
        top5 = 0
        ntotal=0
        dummy_writer = -1
        for iteration,batch in enumerate(self.testing_data_loader,1):
            inputs = batch[0].to(self.device).float()
            target = batch[1].to(self.device).long()
            logits_list,logits = self.model(inputs)
            #logits = self.model(inputs) Uncomment when Using SA-Net

            for n in range(logits.shape[0]):
                with torch.no_grad():
                    if dummy_writer == -1:
                        dummy_writer = target[0]
                    if target[n]!=dummy_writer:
                        dummy_writer = target[n]
                    dummy_pred = logits[n].cpu().numpy()
                    self.page[dummy_writer] += dummy_pred

            

            logist = logits/3 
            res = self.accuracy(logits,target,topk=(1,5))
            top1 += res[0]
            top5 += res[1]
            
            ntotal += inputs.size(0)

        page_acc = 0
        for i in range(self.page.shape[0]):
            writer_page = np.argmax(self.page[i])
            if writer_page==i:
                page_acc +=1

        top1 /= float(ntotal)
        top5 /= float(ntotal)
        page_acc /= float(self.page.shape[0])
        print('Testing on epoch: %d has accuracy: top1: %.2f top5: %.2f'%(epoch,top1*100,top5*100))
        print('Testing pages on epoch: %d has page accuracy: top1: %.2f'%(epoch,page_acc*100))
        with open(self.logfile,'a') as fp:
            fp.write('Testing epoch %d accuracy is: top1: %.2f top5: %.2f\n'%(epoch,top1*100,top5*100))
            fp.write(('Testing pages on epoch: %d has page accuracy: top1: %.2f'%(epoch,page_acc*100)))
    def check_exists(self,epoch):
        model_out_path = self.model_dir + '/' + self.modelfile + '-model_epoch_{}.pth'.format(epoch)
        return os.path.exists(model_out_path)
    
    def checkpoint(self,epoch):
        model_out_path = self.model_dir + '/' + self.modelfile + '-model_epoch_{}.pth'.format(epoch)
        torch.save(self.model.state_dict(),model_out_path)
    
    
    def load_model(self,epoch):
        model_out_path = self.model_dir + '/' + self.modelfile + '-model_epoch_{}.pth'.format(epoch)
        print(model_out_path)
        self.model.load_state_dict(torch.load(model_out_path))
        print('Load model successful')
                
    def train_loops(self,start_epoch,num_epoch):
        #if self.check_exists(num_epoch): return
        if start_epoch > 0:
            self.load_model(start_epoch-1)
        
        for epoch in range(start_epoch,num_epoch):
             
            self.train(epoch)
            self.checkpoint(epoch)
            self.test(epoch)
            self.scheduler.step()
        
    def accuracy(self,output,target,topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            _,pred = output.topk(maxk,1,True,True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            
            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.data.cpu().numpy())
        
        return res
                        
                        
                
                
if __name__ == '__main__':
	
    modelist = ['vertical','horzontal']
    mode = modelist[0]
    print('results on iam  by msrf net')
    mod = DeepWriter_Train(dataset='IAM',mode=mode) # change the dataset to Firemaker or CERUG-EN
    mod.train_loops(0,50)
