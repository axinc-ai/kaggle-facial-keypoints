import dataloader, resnet_model, utils
import torch
import torch.nn as nn
import math
from statistics import mean
from tensorboardX import SummaryWriter


# TODO SAVE and LOAD training model
# TODO parser
# TODO create SummaryWriter()
# TODO log
# TODO gradient clipping ?
# TODO generate images
# TODO test mode
# TODO Cross validation ?
# TODO which optimezer to use ? (For now, adam)


BATCH_SIZE = 128
EPOCH_SIZE = 500
DROPOUT = 0.4
SHUFFLE = False  # TODO normally, True is better when training !
L_RATE = 1e-04  # TODO find best value !
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SAVE_NAME = "save.pt"

print("Create new model")
model = resnet_model.ResNet(dropout=DROPOUT).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), L_RATE)
print(model)


def training():
    train_data_loader = dataloader.DataLoader(BATCH_SIZE, test=False)
    nb_imgs = train_data_loader.nb_file
    # test_data_loader = dataloader.DataLoader(BATCH_SIZE, test=True)  # TODO for now, comment out test mode

    best_loss = math.inf
    init_epoch = 1

    for epoch in range(init_epoch, EPOCH_SIZE + 1):
        print(">>> Epoch {} / {}".format(epoch, EPOCH_SIZE))
        print(">>> Start training")
        model.train()

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
            if iteration % 10 == 0 or iteration == 1:
                utils.save_figures(X, out, "training_images/train_{}_{}.png".format(epoch, iteration // 10))
                print("training loss : {:12.4f}    {:3} %".format(train_loss, 100 * iteration / (nb_imgs / BATCH_SIZE)), end='\r')
        avg_train_loss = mean(train_losses)
        print("\n >>> Average training loss: {}".format(avg_train_loss))
        train_data_loader.restart(shuffle=SHUFFLE)
        if avg_train_loss < best_loss:  # TODO cross-validation ?
            print(">>> Saving models...")
            best_loss = avg_train_loss
            save_dict = {"epoch": epoch,
                         "optimizer": optimizer.state_dict()
                         }
            model.info_dict = save_dict
            torch.save(model, SAVE_NAME)


if __name__ == "__main__":
    training()



