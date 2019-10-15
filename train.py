import dataloader, resnet_model, utils
import torch
import torch.nn as nn
import math
from statistics import mean
import argparse
import sys
from tensorboardX import SummaryWriter


# TODO create SummaryWriter()
# TODO log
# TODO gradient clipping ?
# TODO generate images
# TODO Cross validation ?
# TODO which optimezer to use ? (For now, adam)
# TODO check if this model doesn't just get average


BATCH_SIZE = 256
EPOCH_SIZE = 300
DROPOUT = 0.4
SHUFFLE = False  # TODO normally, True is better when training !
L_RATE = 1e-04  # TODO find best value !
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SAVE_NAME = "save.pt"

# Argument
parser = argparse.ArgumentParser()
parser.add_argument("--load", help='load [save.pt] model', action="store_true")
args = parser.parse_args()

# Load or create model
loading = args.load
if loading:
    try:
        if DEVICE == "cpu":
            model = torch.load(SAVE_NAME, map_location={"cuda:0": "cpu"})
        else:
            model = torch.load(SAVE_NAME, map_location={"cpu": "cuda:0"})
        optimizer = torch.optim.Adam(model.parameters(), L_RATE)
        optimizer.load_state_dict(model.info_dict['optimizer'])
        print("Success loading model")
    except IOError:
        print("Could not find " + SAVE_NAME)
        sys.exit(0)
else:
    print("Create new model")
    model = resnet_model.ResNet(dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), L_RATE)
print(model)


def training():
    train_data_loader = dataloader.DataLoader(BATCH_SIZE, test=False)
    nb_train_imgs = train_data_loader.nb_file
    test_data_loader = dataloader.DataLoader(BATCH_SIZE, test=True)

    if loading:
        best_loss = float(model.info_dict['best_loss'])
        init_epoch = int(model.info_dict['epoch']) + 1
        print("Best loss is {:4f}".format(best_loss))
    else:
        best_loss = math.inf
        init_epoch = 1

    for epoch in range(init_epoch, EPOCH_SIZE + 1):
        print(">>> Epoch {} / {}".format(epoch, EPOCH_SIZE))
        print(">>> Start training")
        model.train()  # training mode

        train_losses = []
        iteration = 0
        while train_data_loader.next_is_available():
            iteration += 1
            X, y = train_data_loader.get_batch()
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(X)
            train_loss = nn.MSELoss()(out, y)
            train_losses.append(train_loss.item())
            train_loss.backward()
            optimizer.step()
            # if iteration % 10 == 0 or iteration == 1:
            #     utils.save_figures(X, out, "training_images/train_{}_{}.png".format(epoch, iteration // 10))
            #     print("training loss : {:12.4f}".format(train_loss), end='\r')
        avg_train_loss = mean(train_losses)
        print("\n >>> Average training loss: {}".format(avg_train_loss))
        train_data_loader.restart(shuffle=SHUFFLE)

        print(">>> Start Test")
        model.eval()  # evaluation mode
        val_iteration = 0
        with torch.no_grad():
            while test_data_loader.next_is_available():
                val_iteration += 1
                X, _ = test_data_loader.get_batch()
                X = X.to(DEVICE)
                out = model(X)
                if val_iteration == 1:
                    utils.save_figures(X, out, "test_images/test_{}.png".format(epoch))
            test_data_loader.restart()

        if avg_train_loss < best_loss:  # TODO cross-validation ?
            print(">>> Saving models...")
            best_loss = avg_train_loss
            save_dict = {"epoch": epoch,
                         "best_loss": best_loss,
                         "optimizer": optimizer.state_dict()
                         }
            model.info_dict = save_dict
            torch.save(model, SAVE_NAME)


if __name__ == "__main__":
    training()



