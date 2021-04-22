import torch
import pandas as pd
from torch.utils.data import DataLoader
from modules.train_scripts import train
import argparse
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from modules.dataset import DataClass
from PIL import Image
from pathlib import Path
from torchvision import transforms
import pickle
from modules.utils import predict
import numpy as np


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='./data')
    parser.add_argument("--result_path", type=str, default='./results/')
    parser.add_argument("--checkpoints_path", type=str, default='./checkpoints/')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model_name', type=str, default='efficientnet-b0')
    parser.add_argument('--image_path', type=str, default='./data/val/n01440764/n01440764_141.JPEG') #TODO add default path
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--experiment_name', type=str, default='default_experiment')
    parser.add_argument('--one_sample', action='store_true')
    args = parser.parse_args()

    DEVICE = torch.device(args.device)

    checkpoint = torch.load(args.checkpoints_path + args.experiment_name + '/model.pt', map_location=torch.device(DEVICE))

    if args.pretrained:
        model = EfficientNet.from_pretrained(args.model_name)
    else:
        model = EfficientNet.from_name(args.model_name)
    model._fc = nn.Linear(in_features=1280, out_features=args.num_classes, bias=True)
    model = model.to(DEVICE)
    model.load_state_dict(checkpoint)

    label_encoder = pickle.load(open(args.checkpoints_path + args.experiment_name + '/label_encoder.pkl', 'rb'))
    if args.one_sample:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        image = Image.open(args.image_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)
        with torch.no_grad():
            image = image.to(DEVICE)
            model.eval()
            logit = model(image).cpu()
            probs = torch.nn.functional.softmax(logit, dim=-1)
        y_pred = torch.argmax(probs, -1).item()
        print(label_encoder.classes_[y_pred])
    else:
        val_dataset = DataClass(args.dataset_path, mode='val')
        files = val_dataset.files
        images = [val_dataset[idx][0].unsqueeze(0) for idx in range(len(val_dataset))]
        probs = predict(model, images, args.device)
        y_pred = np.argmax(probs, -1)
        true_labels = [label_encoder.classes_[val_dataset[idx][1]] for idx in range(len(val_dataset))]
        preds_labels = [label_encoder.classes_[i] for i in y_pred]
        df = pd.DataFrame.from_dict({'img_name': files, 'true': true_labels, 'pred': preds_labels})
        df.to_csv(args.result_path + args.experiment_name + '.csv', index=False)

