import time

import torch
import torch.optim as optim
from models import Vgg16
import torch.nn as nn
from tqdm.notebook import tqdm
from utils import *

def train(epochs, train_loader, test_loader, model, optimizer):
    LOSSES_TRAIN = []
    LOSS_TRACE_FOR_TRAIN = []
    ACCS_VAL = []
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        count = 0
        model.train()
        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            X_train, Y_train = batch
            X_train, Y_train = X_train.to(device), Y_train.to(device)
            Y_pred_train  = model(X_train)
            Y_train = Y_train.squeeze(-1)
            LOSS_train = criterion(Y_pred_train, Y_train)
            LOSS_TRACE_FOR_TRAIN.append(LOSS_train.cpu().detach().numpy())
            LOSS_train.backward()
            optimizer.step()
            count += 1
        LOSSES_TRAIN.append(np.average(LOSS_TRACE_FOR_TRAIN))
        print(f"Epoch : {epoch+1:02}/{epochs} | Train Loss : {np.average(LOSS_TRACE_FOR_TRAIN):.5f}", end = " ")
        test(test_loader, model, criterion)

def test(test_loader, model, criterion):
    with torch.no_grad():
        model.eval()
        Result_pred_val, Result_anno_val = [], []
        LOSS_TRACE_FOR_VAL = []
        for idx, batch in enumerate(test_loader):
            X_val, Y_val = batch
            X_val, Y_val = X_val.to(device), Y_val.to(device)

            Y_pred_val = model(X_val)
            LOSS_val = criterion(Y_pred_val, Y_val)
            LOSS_TRACE_FOR_VAL.append(LOSS_val.cpu().detach().numpy())

            Y_pred_val_np = Y_pred_val.to('cpu').detach().numpy()
            Y_pred_val_np = np.argmax(Y_pred_val_np, axis=1).squeeze()
            Y_val_np = Y_val.to('cpu').detach().numpy().reshape(-1, 1).squeeze()

            Result_pred_val = np.hstack((Result_pred_val, Y_pred_val_np))
            Result_anno_val = np.hstack((Result_anno_val, Y_val_np))
            # Result_pred_val.append(list(Y_pred_val_np))
            # Result_anno_val.append(list(Y_val_np))

        Result_pred_np = np.array(Result_pred_val)
        Result_anno_np = np.array(Result_anno_val)
        Result_pred_np = np.reshape(Result_pred_np, (-1, 1))
        Result_anno_np = np.reshape(Result_anno_np, (-1, 1))

        ACC_VAL = metrics.accuracy_score(Result_anno_np, Result_pred_np)
        LOSS_VAL = np.average(LOSS_TRACE_FOR_VAL)
        print(f"| ACC_VAL : {ACC_VAL*100:.2f} | LOSS_VAL : {LOSS_VAL:.5f}")

def main():
    import torchvision
    SaveModelName = "vgg16"

    ModelSavePath = "./models/" + SaveModelName + "/"

    if not os.path.isdir(ModelSavePath):
        os.mkdir(ModelSavePath)

    TRAIN_SET, TEST_SET, train_loader, test_loader = load_dataset()
    n_dog_breed_classes = 10
    model = torchvision.models.vgg16(pretrained = True)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=n_dog_breed_classes,bias=True)
    # model = Vgg16(n_dog_breed_classes)
    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [40, xxx] -> [10, ...], [10, ...], [10, ...], [10, ...] on 4 GPUs
        model = nn.DataParallel(model)
    # Use appropriate device
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Start Training")
    train(epochs, train_loader, test_loader, model, optimizer)

if __name__ == '__main__':
    main()


