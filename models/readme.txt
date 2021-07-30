Model filenanes indicate maximum length of the description string used for training and number of training epochs.
To avoid optimizer related warnings, load them using tensorflow.keras.models.load_model with compile=False. (We are not using the optimizer for this anyway).
