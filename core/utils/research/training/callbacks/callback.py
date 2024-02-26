class Callback:
    def on_epoch_start(self, model, epoch, logs=None):
        pass

    def on_epoch_end(self, model, epoch, losses, logs=None):
        pass

    def on_batch_start(self, model, batch, logs=None):
        pass

    def on_batch_end(self, model, batch, logs=None):
        pass

    def on_train_start(self, model, logs=None):
        pass

    def on_train_end(self, model, logs=None):
        pass
