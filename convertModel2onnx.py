import torch
import sys
from torch.autograd import Variable


def torch2onnx(filename, image_size):
    try:
        model = torch.load(filename, map_location={"cuda:0": "cpu"})
    except IOError:
        print("Could not find " + filename)
        sys.exit(0)

    model.train(False)
    dummy = Variable(torch.randn(1, 1, image_size, image_size))
    torch.onnx.export(
        model, dummy, 'resnet_facial_feature.onnx',
        verbose=True, opset_version=10
    )
    print("Export is done")


if __name__ == "__main__":
    filename = input("PyTorch model save file name : ")
    image_size = int(input("96 | 226 : "))
    torch2onnx(filename, image_size)
