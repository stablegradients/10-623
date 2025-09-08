import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import argparse
import wandb

img_size = (256,256)
num_labels = 3
normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class CsvImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if idx >= self.__len__(): raise IndexError()
        img_name = self.data_frame.loc[idx, "image"]
        image = Image.open(img_name).convert("RGB")  # Assuming RGB images
        label = self.data_frame.loc[idx, "label_idx"]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data(args):
    transform_img = T.Compose([
        T.ToTensor(), 
        T.Resize(min(img_size[0], img_size[1]), antialias=True),  # Resize the smallest side to 256 pixels
        T.CenterCrop(img_size),  # Center crop to 256x256
        T.Normalize(mean=normalize_mean, std=normalize_std), # Normalize each color dimension
        ])
    
    if args.grayscale:
        # Append grayscale transformation
        transform_img = T.Compose([
            T.ToTensor(), 
            T.Grayscale(1),
            T.Resize(min(img_size[0], img_size[1]), antialias=True),  # Resize the smallest side to 256 pixels
            T.CenterCrop(img_size),  # Center crop to 256x256
            T.Normalize(mean=normalize_mean, std=normalize_std), # Normalize each color dimension
        ])
        
    train_data = CsvImageDataset(
        csv_file='./data/img_train.csv',
        transform=transform_img,
    )
    test_data = CsvImageDataset(
        csv_file='./data/img_test.csv',
        transform=transform_img,
    )
    val_data = CsvImageDataset(
        csv_file='./data/img_val.csv',
        transform=transform_img,
    )
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size)

    for X, y in train_dataloader:
        print(f"Shape of X [B, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    
    return train_dataloader, test_dataloader, val_dataloader

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # First layer input size must be the dimension of the image
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(img_size[0] * img_size[1] * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, 4, 4)
        self.layernorm1 = nn.LayerNorm([128, 64, 64])
        self.conv2 = nn.Conv2d(128, 128, 7, 1, 3)
        self.layernorm2 = nn.LayerNorm([128, 64, 64])
        self.conv3 = nn.Conv2d(128, 256, 1, 1)
        self.act1 = nn.GELU()
        self.conv4 = nn.Conv2d(256, 128, 1, 1)
        self.twod_avg_pool = nn.AvgPool2d((2, 2))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128*32*32, num_labels)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.layernorm1(x)
        x = self.conv2(x)
        x = self.layernorm2(x)
        x = self.conv3(x)
        x = self.act1(x)
        x = self.conv4(x)
        x = self.twod_avg_pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

def denormalize(x):
    denom_transform = T.Normalize(
        mean=[-m/s for m, s in zip(normalize_mean, normalize_std)], 
        std=[1/s for s in normalize_std]
    )
    return torch.clamp(denom_transform(x), 0, 1)

def train_one_epoch(dataloader, model, loss_fn, optimizer, t):
    size = len(dataloader.dataset)
    total_steps_this_epoch = 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_size = len(y)
        loss = loss.item() / batch_size
        current = (batch + 1) * batch_size
        total_steps_this_epoch += batch_size
        wandb.log({"train_loss": loss, "num_train_examples": t*size + total_steps_this_epoch})
        if batch % 10 == 0:
            print(f"Train batch avg loss = {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
    
def evaluate(dataloader, dataname, model, loss_fn, is_last_epoch=False, tag=''):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    avg_loss, correct = 0, 0
    step_counter = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            avg_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            step_counter += 1
            images_to_log = []
            if is_last_epoch and step_counter == 1:
                for i in range(len(X)):
                    label_idx_map = {0: "parrot", 1: "narwhal", 2: "axolotl"}
                    caption = str(label_idx_map[pred[i].argmax(0).item()]) +  " / " + str(label_idx_map[y[i].item()])
                    images_to_log.append(wandb.Image(denormalize(X[i]), caption=caption, mode="RGB"))
                wandb.log({"media_" + tag + "/": images_to_log})
    avg_loss /= size
    correct /= size
    print(f"{dataname} accuracy = {(100*correct):>0.1f}%, {dataname} avg loss = {avg_loss:>8f}")
    return correct, avg_loss

def main(args):
    torch.manual_seed(10999)
    
    if args.use_wandb:
        # get the hyperparameters dict
        hyperparameters = {
            "model": args.model,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "n_epochs": args.n_epochs,
            "grayscale": args.grayscale,
            "use_wandb": args.use_wandb,
        }
        wandb.init(entity="stablegradients", project="hw0_img_classifier_shrinivr", name="neural-the-narwhal-original", config=hyperparameters)
    else:
        wandb.init(mode='disabled')
    
    print(f"Using {device} device")
    train_dataloader, test_dataloader, val_dataloader = get_data(args)
    
    if args.model == 'simple':
        model = NeuralNetwork().to(device)
    elif args.model == 'cnn':
        model = CNN().to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    print(model)
    # print model size
    print("Model size: ", sum(p.numel() for p in model.parameters()))
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    for t in range(args.n_epochs):
        print(f"\nEpoch {t+1}\n-------------------------------")
        train_one_epoch(train_dataloader, model, loss_fn, optimizer, t) 
        if t == args.n_epochs - 1:
            is_last_epoch = True
        else:
            is_last_epoch = False
        train_correct, train_loss = evaluate(train_dataloader, "Train", model, loss_fn, is_last_epoch, tag="train")
        test_correct, test_loss = evaluate(test_dataloader, "Test", model, loss_fn, is_last_epoch, tag="test")
        val_correct, val_loss = evaluate(val_dataloader, "Val", model, loss_fn, is_last_epoch, tag="val")
        wandb.log({"train_accuracy_epoch": train_correct, "train_loss_epoch": train_loss, 
                   "val_accuracy_epoch": val_correct, "val_loss_epoch": val_loss, 
                   "test_accuracy_epoch": test_correct, "test_loss_epoch": test_loss, "epoch": t+1})
    
    print("Done!")

    # Save the model
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

    # Load the model (just for the sake of example)
    if args.model == 'simple':
        model = NeuralNetwork().to(device)
    elif args.model == 'cnn':
        model = CNN().to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    model.load_state_dict(torch.load("model.pth", weights_only=True))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Image Classifier')
    parser.add_argument('--use_wandb', action='store_true', default=False, help='Use Weights and Biases for logging')
    parser.add_argument('--n_epochs', type=int, default=5, help='The number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='The batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='The learning rate for the optimizer')
    parser.add_argument('--model', type=str, choices=['simple', 'cnn'], default='simple', help='The model type')
    parser.add_argument('--grayscale', action='store_true', default=False, help='Use grayscale images instead of RGB')
    
    args = parser.parse_args()
    
    main(args)