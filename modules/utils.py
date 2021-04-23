import torch.nn as nn
from math import pi, cos
import torch
from tqdm import tqdm

def weights_init(m):
    """
    Kaiming uniform weight initialization
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)


def warm_up_and_annealing(it, warm_up_epochs, epochs, dataloader, resume_training):
    """
    linearly warms up learning rate within warm_up_epochs period
    and anneals it afterwars using cosineLr function
    
    :param it:{int} batch iteration
    :param warm_up_epochs:{int} number of epochs to warm up learning rate 
    :param epochs: {int} total number of epochs
    :param dataloader: dataloader
    :param resume_training:{bool} True if training is resumed
    :return :{float} scaling learning rate with scheduler lambdaLR
    """
    m = warm_up_epochs * len(dataloader)
    if it < m and not resume_training:
        return it / m
    else:
        T = (epochs - warm_up_epochs) * (len(dataloader)) if epochs != warm_up_epochs else 1
        return 1 / 2 * (1 + cos(it * pi / T))


def predict(model, test_loader, device):
    """
    predicting probs on more then one sample of data
    used when on_sample in predict.py is not activated
    
    :param model: neural net
    :param test_loader: data loader
    :param resume_training:{str} cuda or cpu 
    :return :{np.arrray} array with probs distribution for each sample 
    """
    with torch.no_grad():
        logits = []
        for inputs in tqdm(test_loader):
            inputs = inputs.to(device)
            model.eval()
            outputs = model(inputs).cpu()
            logits.append(outputs)
    probs = nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
    return probs
