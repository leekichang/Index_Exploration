import torchvision.models
import models
from utils import *
import torch.nn as nn
import torch.optim as optim
import time

def train(FeatureExtractor, classifier, decoder, train_loader, optim_c, optim_d):
    classifier.train()
    decoder.train()
    classifier_loss = nn.CrossEntropyLoss().to(device)
    batch_time = ExpoAverageMeter()  # forward prop. + back prop. time
    losses = ExpoAverageMeter()  # loss (per word decoded)
    start = time.time()
    for idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        optim_c.zero_grad()
        optim_d.zero_grad()

        feature = FeatureExtractor(x)
        pred_c = classifier(feature)
        pred_d = decoder(feature)

        loss_c = classifier_loss(pred_c, y)
        loss_d = torch.sqrt((pred_d - x).pow(2).mean())
        loss_c.backward()
        loss_d.backward()

        optim_c.step()
        optim_d.step()

        losses.update(loss_d.item())
        batch_time.update(time.time()-start)

        start = time.time()

def main():
    FeatureExtractor = models.Vgg16FeatureExtractor()
    Classifier = models.VggClassifier(n_class=10)
    SegNet_Decoder = models.SegNet_Decoder()

    train_set, train_loader, test_set, test_loader = load_dataset()
    train(FeatureExtractor, Classifier, SegNet_Decoder, train_loader)

if __name__ == "__main__":
    main()