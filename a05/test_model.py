import torch
import pytest
from model import MNISTModel
from train import train_model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_parameter_count():
    model = MNISTModel()
    param_count = count_parameters(model)
    print("Model has param_count " +  str(param_count))

def test_model_accuracy():
    _, accuracy = train_model()
    print("Model accuracy is" + str(accuracy))

test_parameter_count()
test_model_accuracy()