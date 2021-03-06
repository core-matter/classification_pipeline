num_classes                         = 10                        # Number of classes.
num_epochs                          = 10                        # Total number of epochs.
minibatch_size                      = 32                        # Number of samples in batch.
batch_normalization_momentum        = 0.999                     # Batch normalization momentum.
learning_rate_max                   = 0.003                     # Maximum learning rate.
augment_translation                 = 0.0625                    # Image translation by augment_translation * img_size pixels
start_epoch                         = 0                         # Which epoch to start training from. For continuing a previously trained network.
load_network_filename               = None                      # Set to load a previously saved network.
warm_up_epochs                      = 1                         # number of epochs to warm up learning rate
