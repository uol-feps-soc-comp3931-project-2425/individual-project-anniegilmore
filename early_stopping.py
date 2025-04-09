class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.lowest_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.lowest_loss is None:
            self.lowest_loss = val_loss
        elif val_loss > self.lowest_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.lowest_loss = val_loss
            self.counter = 0
