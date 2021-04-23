# classification_pipeline
Pipline for classification task in CV on PyTorch.


The pipline leverages collection of Efficient-nets (b0-b7) and 
tested on [Imagenette dataset](https://github.com/fastai/imagenette). 

```
Repository
│   README.md
│   requierements.txt    
│   train.py
|   predict.py
|   example of usage.ipynb
└───modules
    │   config.py
    │   dataset.py
    │   utils.py
    |   train_scripts.py
    
```

Structure of data is supposed to be as follows:

```
data_folder_name
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
train.py file serves as a facade of the pipeline and as such
the file incorporates necessary parameters listed below.  
~~~
 train.py [-h]  [--dataset_path DATASET_PATH]
                [--checkpoints_path CHECKPOINTS_PATH]
                [--writer_path WRITER_PATH] [--lr LR] [--epochs EPOCHS]
                [--warm_up_epochs WARM_UP_EPOCHS] [--num_classes NUM_CLASSES]
                [--batch_size BATCH_SIZE] [--start_epoch START_EPOCH]
                [--num_workers NUM_WORKERS] [--device DEVICE]
                [--model_name MODEL_NAME] [--pretrained PRETRAINED]
                [--experiment_name EXPERIMENT_NAME]
                [--resume_training RESUME_TRAINING]
~~~
## Prediciton
Prediciton file parameters.

Note if one sample parameter is set(no need to specify true or false)
prediction returns 'class_name_folder' name
otherwise it saves csv file with true labels and predicted labels
set correspondently.
~~~
predict.py [-h]   [--checkpoints_path CHECKPOINTS_PATH] [--device DEVICE]
                  [--model_name MODEL_NAME] [--image_path IMAGE_PATH]
                  [--pretrained PRETRAINED] [--num_classes NUM_CLASSES]
                  [--experiment_name EXPERIMENT_NAME]
                  [--one_sample ONE_SAMPLE]
 
~~~ 
## [Results](https://tensorboard.dev/experiment/9LbZmsE5Tri4TGudWV5MFg/#scalars) of efficientnet-b0 and example of usage
~~~

For interaction with the pipline use 'example of usage.ipynb' file

~~~
## TODO:
~~~
-add label smoothing option
-add help comments in argparse
