import dataloader, resnet_model, utils
import torch
import torch.nn as nn
import math
from statistics import mean
import argparse
import sys
from tensorboardX import SummaryWriter
from datetime import datetime


# TODO create SummaryWriter()
# TODO log
# TODO gradient clipping ?
# TODO generate images
# TODO Cross validation ?
# TODO which optimezer to use ? (For now, adam)
# TODO check if this model doesn't just get average
# TODO VARIABLES should be arguments

"""
python3 train.py --load
"""

dt = str(datetime.now()).replace(" ", "_")

BATCH_SIZE = 256
EPOCH_SIZE = 1000
SAVE_EPOCH_LIST = [300, 500, 800]  # save model separtely
DROPOUT = 0.4
SHUFFLE = False  # TODO normally, True is better when training !
L_RATE = 1e-04  # TODO find best value !
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SAVE_NAME = "checkpoints/save.pt"
SUMMARY_WRITER_PATH = "runs/flip_and_rotate30" + dt

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
    eval_X, eval_y = train_data_loader.get_eval_data()
    eval_data_loader = dataloader.DataLoader(
        BATCH_SIZE, test=False, X=eval_X, y=eval_y
    )

    writer = SummaryWriter(SUMMARY_WRITER_PATH)

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
            if iteration == 1:
                utils.save_figures(
                    X,
                    out,
                    "training_images/train_{}.png".format(epoch)
                )
                print("training loss : {:12.4f}".format(train_loss), end='\r')
        avg_train_loss = mean(train_losses)
        print("\n >>> Average training loss: {}".format(avg_train_loss))
        writer.add_scalar('avg_train_loss', avg_train_loss, epoch)
        train_data_loader.restart(shuffle=SHUFFLE)

        print(">>> Start Test")
        model.eval()  # evaluation mode
        test_losses = []
        val_iteration = 0
        with torch.no_grad():
            while eval_data_loader.next_is_available():
                val_iteration += 1
                X, y = eval_data_loader.get_batch()
                X, y = X.to(DEVICE), y.to(DEVICE)
                out = model(X)
                test_loss = nn.MSELoss()(out, y)
                test_losses.append(test_loss.item())
                if val_iteration == 1:
                    utils.save_figures(
                        X,
                        out,
                        "test_images/test_{}.png".format(epoch))
            avg_test_loss = mean(test_losses)
            print("\n >>> Average test loss: {}".format(avg_test_loss))
            writer.add_scalar('avg_test_loss', avg_test_loss, epoch)
            eval_data_loader.restart()

        # TODO SAVE_EPOCH_LIST should just rename the existing save.pt
        if avg_test_loss < best_loss or epoch in SAVE_EPOCH_LIST:
            print(">>> Saving models...")
            best_loss = avg_test_loss
            save_dict = {"epoch": epoch,
                         "best_loss": best_loss,
                         "optimizer": optimizer.state_dict()
                         }
            model.info_dict = save_dict
            if epoch in SAVE_EPOCH_LIST:
                torch.save(
                    model,
                    SAVE_NAME.replace(".pt", "_{}.pt".format(epoch))
                )

            else:
                torch.save(model, SAVE_NAME)

    # writer.export_scalars_to_json("./all_scalars.json")  # Do we need this ?
    writer.close()


if __name__ == "__main__":
    training()
