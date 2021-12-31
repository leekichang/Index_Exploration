import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from utils import *

vgg16 = models.vgg16(pretrained=True)

vgg16.classifier[6] = nn.Linear(in_features=4096, out_features=10, bias=True)

TRAIN_SET, TEST_SET, train_loader, test_loader = load_dataset()
optimizer = optim.Adam(vgg16.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

LOSSES_TRAIN=[]
LOSS_TRACE_FOR_TRAIN=[]

print("Start Training")
for epoch in range(10):
    vgg16.train()
    for idx, batch in enumerate(train_loader):
        print("Check")
        optimizer.zero_grad()
        X_train, Y_train = batch
        X_train, Y_train = X_train.to(device), Y_train.to(device)
        Y_pred_train = vgg16(X_train)
        Y_train = Y_train.squeeze(-1)
        LOSS_train = criterion(Y_pred_train, Y_train)
        LOSS_TRACE_FOR_TRAIN.append(LOSS_train.cpu().detach().numpy())
        LOSS_train.backward()
        optimizer.step()
    LOSSES_TRAIN.append(np.average(LOSS_TRACE_FOR_TRAIN))
    print(f"Epoch : {epoch + 1:02} | Train Loss : {np.average(LOSS_TRACE_FOR_TRAIN):.5f}")
