import sys
import torch
import utils, dataloader


# generate submissions.csv file
def inference():
    save_file = input("save model name : ")
    try:
        model = torch.load(save_file, map_location={"cpu": "cuda:0"})
        print("Success loading model")
    except IOError:
        print("Couldn't find model")
        sys.exit(0)

    test_data_loader = dataloader.DataLoader(1783, test=True)  # 1783 : length of test data set
    model.eval()
    with torch.no_grad():
        X, _ = test_data_loader.get_batch()
        X = X.to("cuda:0")
        output = model(X)
    utils.generate_csv(output)


if __name__ == "__main__":
    inference()
