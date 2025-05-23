import torch
import torch.nn as nn


def get_input_saliency(model: nn.Module, X: torch.Tensor):

	X = X.clone().detach().requires_grad_(True)
	y_hat = model(X)
	y_hat.backward(torch.ones_like(y_hat))
	saliency = torch.mean(X.grad.abs(), dim=0)

	return saliency
