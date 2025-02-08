from torch.utils.data import DataLoader


class SimpleTrainer:

	def __init__(self, model):
		self.model = model
		self.cls_loss_function = None
		self.reg_loss_function = None
		self.optimizer = None

	def __split_y(self, y):
		cls_y = y[:, :1]
		reg_y = y[:, 1:]
		return cls_y, reg_y

	def __loss(self, y_hat, y):
		cls_y, reg_y = self.__split_y(y)
		cls_y_hat, reg_y_hat = self.__split_y(y_hat)

		cls_loss = self.cls_loss_function(cls_y_hat, cls_y)
		reg_loss = self.reg_loss_function(reg_y_hat, reg_y)

		loss = cls_loss + reg_loss
		return cls_loss, reg_loss, loss

	def train(self, dataloader: DataLoader, epochs: int):

		dataloader_size = len(dataloader)

		for epoch in range(epochs):

			for i, (X, y) in enumerate(dataloader):
				self.optimizer.zero_grad()
				y_hat = self.model(X)
				cls_loss, reg_loss, loss = self.__loss(y_hat, y)
				loss.backward()
				self.optimizer.step()

				if i % 10 == 0:
					print(f"Epoch: {epoch}/{epochs}, Batch: {i}/{dataloader_size}, Loss: {loss}", end="\r")
