import sys
import torch
import utils, dataloader


# generate submissions.csv file
def inference():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    save_file = input("save model name : ")
    try:
        if torch.cuda.is_available():
            model = torch.load(save_file, map_location={"cpu": "cuda:0"})
        else:
            model = torch.load(save_file, map_location={"cuda:0": "cpu"})
        print("Success loading model")
    except IOError:
        print("Couldn't find model")
        sys.exit(0)

    # 1783 : length of test data set
    test_data_loader = dataloader.DataLoader(1783, test=True)  
    model.eval()
    with torch.no_grad():
        X, _ = test_data_loader.get_batch()
        X = X.to(device)
        output = model(X)
    utils.generate_csv(output)


if __name__ == "__main__":
    inference()
