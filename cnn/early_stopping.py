class EarlyStopping:
    """Class to implement overfitting defense, early stopping

    Args:
        patience (int): defines the number of epochs for which stagnating or increase val loss is tolerated

    """

    def __init__(self, patience=3):
        """
        Args:
            counter (int): counts number of epochs for which val loss has stagnated or increased
            lowest_validation_loss (float): records the lowest (best) validation loss reached by the model
            early_stop (bool): indicates whether the model training should be interrupted

        """
        self.patience = patience
        self.counter = 0
        self.lowest_validation_loss = None
        self.early_stop = False

    def __call__(self, validation_loss: float):
        """
        Args:
            validation_loss (float): the most recent validation loss achieved by the model

        """
        if self.lowest_validation_loss is None:
            self.lowest_validation_loss = validation_loss
        elif validation_loss > self.lowest_validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.lowest_validation_loss = validation_loss
            self.counter = 0
