import time

import torch
import torch.optim as optim
from models import Vgg16
import torch.nn as nn
from tqdm.notebook import tqdm
from utils import *

BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 20

SaveModelName = "vgg16"

ModelSavePath = "./models/" + SaveModelName + "/"
if not os.path.isdir(ModelSavePath):
    os.mkdir(ModelSavePath)


def train(epochs, train_loader, model, optimizer):
    LOSSES_TRAIN = []
    LOSS_TRACE_FOR_TRAIN= []
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        count = 0
        for idx, batch in enumerate(train_loader):
            if idx % 10 == 0:
                print(f'{idx}/256')
            optimizer.zero_grad()
            print(1)
            X_train, Y_train = batch
            print(2)
            X_train, Y_train = X_train.to(device), Y_train.to(device)
            print(3)
            Y_pred_train, indices_1, indices_2, indices_3, indices_4, indices_5 = model(X_train)
            print(4)
            Y_train = Y_train.squeeze(-1)
            print(5)
            LOSS_train = criterion(Y_pred_train, Y_train)
            print(6)
            LOSS_TRACE_FOR_TRAIN.append(LOSS_train.cpu().detach().numpy())
            print(7)
            LOSS_train.backward()
            print(8)
            optimizer.step()
            print(f"step : {count}")
            count += 1
        LOSSES_TRAIN.append(np.average(LOSS_TRACE_FOR_TRAIN))
        print(f"Epoch : {epoch+1:02} | Train Loss : {np.average(LOSS_TRACE_FOR_TRAIN)*100:.2f}%")

def main():
    TRAIN_SET, TEST_SET, train_loader, test_loader = load_dataset()
    n_dog_breed_classes = 10
    model = Vgg16(n_dog_breed_classes)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [40, xxx] -> [10, ...], [10, ...], [10, ...], [10, ...] on 4 GPUs
        model = nn.DataParallel(model)
    # Use appropriate device
    model = model.to(device)

    epochs = 10

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = 100000
    epochs_since_improvement = 0
    print("Start Training")
    train(epochs, train_loader, model, optimizer)

if __name__ == '__main__':
    main()


