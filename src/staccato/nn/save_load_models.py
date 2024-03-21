"""https://pytorch.org/tutorials/beginner/saving_loading_models.html"""
import torch


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(ModelClass, path, model_kwargs={}):
    model = ModelClass(**model_kwargs)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
