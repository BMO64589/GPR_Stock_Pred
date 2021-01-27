# Used for setting devices
import gpytorch
from gpytorch.models import ExactGP, IndependentModelList
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood, LikelihoodList
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal

# PyTorch
import torch

# Timing and math
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
import yfinance as yf
pd.set_option('display.max_columns', 10)

STOCK_NAMES = ["MSFT","AAPL","TSLA", "CRM", "GOOGL","TWT"]
DF = yf.download("MSFT AAPL TSLA CRM GOOGL TWTR", start="2019-01-01", end="2019-04-30")
print(DF.keys())

y = DF["Open"].values.T
x = DF[["Close"]].values.T

nsplit = int(y.shape[1] * .8)
xtrain,ytrain = x[:, :nsplit], y[:, :nsplit]
xtest,ytest = x[:, nsplit:], y[:, nsplit:]

class BatchedGP(gpytorch.models.ExactGP):
    """Class for creating batched Gaussian Process models.  Ideal candidate if
    using GPU-based acceleration such as CUDA for training."""
    def __init__(self, train_x, train_y, likelihood, shape, output_device):
        super(BatchedGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([shape]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(batch_shape=torch.Size([shape])),
            batch_shape=torch.Size([shape]),
            output_device=output_device
        )

    def forward(self, x):
        """Forward pass method for making predictions through the model.  The
        mean and covariance are each computed to produce a MV distribution.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



def train_gp_batched_scalar(Zs, Ys, use_cuda=False, epochs=10, lr=0.1, thr=0):
    """Computes a Gaussian Process object using GPyTorch. Each outcome is
    modeled as a single scalar outcome.

    Parameters:
        Zs (np.array): Array of inputs of expanded shape (B, N, XD), where B is
            the size of the minibatch, N is the number of data points in each
            GP (the number of neighbors we consider in IER), and XD is the
            dimensionality of the state-action space of the environment.
        Ys (np.array): Array of predicted values of shape (B, N, YD), where B is the
            size of the minibatch and N is the number of data points in each
            GP (the number of neighbors we consider in IER), and YD is the
            dimensionality of the state-reward space of the environment.
        use_cuda (bool): Whether to use CUDA for GPU acceleration with PyTorch
            during the optimization step.  Defaults to False.
        epochs (int):  The number of epochs to train the batched GPs over.
            Defaults to 10.
        lr (float):  The learning rate to use for the Adam optimizer to train
            the batched GPs.
        thr (float):  The mll threshold at which to stop training.  Defaults to 0.

    Returns:
        model (BatchedGP): A GPR model of BatchedGP type with which to generate
            synthetic predictions of rewards and next states.
        likelihood (GaussianLikelihood): A likelihood object used for training
            and predicting samples with the BatchedGP model.
    """
    # Preprocess batch data
    B, N = Zs.shape
    print("B: {}".format(B))
    batch_shape = B

    # Format the training features - tile and reshape
    train_x = torch.unsqueeze(torch.tensor(Zs), -1)

    # Format the training labels - reshape
    train_y = torch.tensor(Ys)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([batch_shape]))
    output_device = torch.device('cuda:0')  # GPU
    model = BatchedGP(train_x, train_y, likelihood, batch_shape, output_device)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    if use_cuda:  # Send everything to GPU for training
        model = model.cuda()
        likelihood = likelihood.cuda()
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        mll = mll.cuda()

    def epoch_train(j):
        """Helper function for running training in the optimization loop."""
        optimizer.zero_grad()  # Zero gradients
        output = model(train_x)  # Forwardpass
        loss = -mll(output, train_y).sum()  # Compute ind. losses + aggregate
        loss.backward()  # Backprop gradients
        item_loss = loss.item()
        if j % 5 == 0:
            print("LOSS: {}".format(item_loss))
        optimizer.step()  # Update weights
        optimizer.zero_grad()  # Zero gradients
        gc.collect()  # Remove unnecessary info
        return item_loss

    # Run the optimization loop
    for i in range(epochs):
        loss_i = epoch_train(i)
        if loss_i < thr:  # If we reach a certain loss threshold, stop training
            break

    # Empty the cache from GPU
    torch.cuda.empty_cache()

    return model, likelihood

# GPyTorch training
USE_CUDA = torch.cuda.is_available()
epochs = 1000
thr = -5000
lr = .5
model, likelihood = train_gp_batched_scalar(xtrain, ytrain, use_cuda=USE_CUDA, epochs=epochs, lr=lr, thr=thr)
model.eval()
likelihood.eval()

# Inference/evaluation

# Define mll
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Now evaluate
xtest = torch.unsqueeze(torch.tensor(xtest), -1)
if USE_CUDA:
    xtest = xtest.cuda()

with torch.no_grad():
    observed_pred = likelihood(model(xtest))
    mean = observed_pred.mean

# Get predictions
pred = mean.cpu().numpy()

for i in range(6):
    plt.plot(pred[i, :], color="r", label="Predicted")
    plt.plot(ytest[i, :], color="b", label="True")
    plt.legend()
    plt.xlabel("Time (Days)")
    plt.title("Predicted Closing Prices For {}".format(STOCK_NAMES[i]))
    plt.ylabel("Closing Price")
    plt.show()





