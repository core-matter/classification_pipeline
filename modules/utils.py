import torch.nn as nn
from math import pi, cos
import torch
from tqdm import tqdm

def weights_init(m):
    """
    :param m: Kaiming uniform weight initialization
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)


def warm_up_and_annealing(it, warm_up_epochs, epochs, dataloader, resume_training):
    m = warm_up_epochs * len(dataloader)
    if it < m and not resume_training:
        return it / m
    else:
        T = (epochs - warm_up_epochs) * (len(dataloader)) if epochs != warm_up_epochs else 1
        return 1 / 2 * (1 + cos(it * pi / T))


def predict(model, test_loader, device):
    with torch.no_grad():
        logits = []
        for inputs in tqdm(test_loader):
            inputs = inputs.to(device)
            model.eval()
            outputs = model(inputs).cpu()
            logits.append(outputs)
    probs = nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
    return probs
