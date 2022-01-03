import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_nbr = 3
width = 480
height = 360
batch_size = 22
lr = 0.0001
patience = 50
start_epoch = 0
epochs = 120
print_freq = 4
save_folder = './models'