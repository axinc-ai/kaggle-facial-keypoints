import torch
import sys
from torch.autograd import Variable

def torch2onnx(filename):
    try:
        model = torch.load(filename, map_location={"cuda:0": "cpu"})
    except IOError:
        print("Could not find " + filename)
        sys.exit(0)

    model.train(False)
    dummy = Variable(torch.randn(1, 1, 96, 96))
    torch.onnx.export(model, dummy, 'resnet_facial_feature.onnx', verbose=True, opset_version=10)
    print("Export is done")


if __name__ == "__main__":
    filename = "save_100epoch.pt"
    torch2onnx(filename)
