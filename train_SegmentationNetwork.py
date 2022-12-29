import random

import torch
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm

from dataset import DataLoader
from segmentationFCT import FCT
from segmentationUNet import UNet
from metric import DiceLoss, JaccardScore
from tensorboardX import SummaryWriter 


class Trainer():

    def __init__(self, model_name, pretrained=False) -> None:
        self.model_name = model_name
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.network = self.load_network(pretrained)
        self.batch_size = 8

    def load_network(self, pretrained):
        if self.model_name == "focusnet":
            model = FCT()
            if pretrained:
                check = torch.load('saved_model/model_focusnet.tar')
                model.load_state_dict(check['model_state_dict'])
        else:
            model = UNet(in_channels=3, out_channels=1)
            if pretrained:
                check = torch.load('saved_model/road_model_unet.tar')
                model.load_state_dict(check['model_state_dict'])
        model = model.to(self.device)
        return model
    

    def save_sample(self, epoch, x, y, y_pred):
        path = f'Training Sneakpeeks/unet/{epoch}'
        elements = [x, y, y_pred]
        elements = [transforms.ToPILImage()(torch.squeeze(element[0:1, :, :, :])) for element in elements]

        elements[0] = elements[0].save(f"{path}_input.jpg")
        elements[1] = elements[1].save(f"{path}_actual.jpg")
        elements[2] = elements[2].save(f"{path}_predicted.jpg")


    def train(self, epochs, lr=0.0001):
        print("Loading Datasets...")
        train_dataloader, test_dataloader = DataLoader().load_data(self.batch_size)
        print("Dataset Loaded... initializing parameters...")
        model = self.network
        optimizer = optim.AdamW(model.parameters(), lr)
        # crossentloss = torch.nn.CrossEntropyLoss() 
        # bceloss = torch.nn.BCEWithLogitsLoss() 
        dsc_loss = DiceLoss()  
        # iou = JaccardScore()
        writer = SummaryWriter(log_dir="logs")        
        loss_train, loss_test = [], []
        start = 1
        epochs = epochs+1
        print(f"Starting to train for {epochs} epochs.")

        for epoch in range(start, epochs):
            _loss_train, _loss_test, _measure = 0, 0, 0
            print(f"Training... at Epoch no: {epoch}")
            num = random.randint(0, 300)
            for i, (x, y) in enumerate(tqdm(train_dataloader)):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                y_pred = model(x)
                loss1 = dsc_loss(y_pred, y)
                # loss2 = bceloss(y_pred, y)
                loss = loss1
                # loss = crossentloss(y_pred, y)
                _loss_train += loss.item()
                loss.backward()
                optimizer.step()
                if i == num:
                    self.save_sample(epoch, x, y, y_pred)

            
            # if epoch%20 == 0:
            #     print(f'Evaluating the performace of {epoch} epoch.')
            #     for i, (x, y) in enumerate(tqdm(test_dataloader)):
            #         x, y = x.to(self.device), y.to(self.device)
            #         y_pred = model(x)
            #         # loss = crossentloss(y_pred, y)
            #         loss = dsc_loss(y_pred, y)
            #         measure = iou(y_pred, y)
            #         _measure += measure.item()
            #         _loss_test += loss.item()

            #     writer.add_scalar("Testing Loss", _loss_test, epoch)

            writer.add_scalar("Training Loss", _loss_train, epoch)
            
            loss_train.append(_loss_train)
            loss_test.append(_loss_test)

            print(f"Epoch: {epoch+1}, Training loss: {_loss_train}, Testing Loss: {_loss_test} || Jaccard Score : {_measure}")

            if loss_train[-1] == min(loss_train):
                print('Saving Model...')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_train
                }, f'saved_model/road_model_unet.tar')
            print('\nProceeding to the next epoch...')


    

model = "unet"
seg = Trainer(model, pretrained=False)
seg.train(epochs=70) 
