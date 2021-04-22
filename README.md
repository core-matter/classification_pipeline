# classification_pipeline
Pipline for classification task in CV .


The pipline leverages collection of Efficientnets (b0-b7) and 
tested on [Imagenette dataset](https://github.com/fastai/imagenette)

```
Repository
│   README.md
│   requierements.txt    
│   train.py
|   predict.py
└───modules
    │   config.py
    │   dataset.py
    │   utils.py
    |   train_scripts.py
    
```

Structure of data is supposed to be as follows:

```
data
│
└───train
│   │  
│   └───class_name_folder1
│   |   │    img
│   |   │   ...
│   |       
|   └───class_name_folder2
|        |   img
|        |   ...
|        ...
└───val
   │  
   └───class_name_folder1
   |   │     img
   |   │   ...
   |       
   └───class_name_folder2
        |    img
        |   ...
         ...

```

## Dependencies
~~~~
pip install -r requirements.txt
~~~~
## Training
~~~
 train.py [-h]  [--dataset_path DATASET_PATH]
                [--checkpoints_path CHECKPOINTS_PATH]
                [--writer_path WRITER_PATH] [--lr LR] [--epochs EPOCHS]
                [--warm_up_epochs WARM_UP_EPOCHS] [--num_classes NUM_CLASSES]
                [--batch_size BATCH_SIZE] [--start_epoch START_EPOCH]
                [--supervised_ratio SUPERVISED_RATIO]
                [--num_workers NUM_WORKERS] [--device DEVICE]
                [--model_name MODEL_NAME] [--pretrained PRETRAINED]
                [--experiment_name EXPERIMENT_NAME]
                [--resume_training RESUME_TRAINING]
~~~
## Prediciton
~~~
predict.py [-h]   [--checkpoints_path CHECKPOINTS_PATH] [--device DEVICE]
                  [--model_name MODEL_NAME] [--image_path IMAGE_PATH]
                  [--pretrained PRETRAINED] [--num_classes NUM_CLASSES]
                  [--experiment_name EXPERIMENT_NAME]
                  [--one_sample ONE_SAMPLE]
