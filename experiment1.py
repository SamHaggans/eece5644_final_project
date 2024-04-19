import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from scipy import stats
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import ClippedAdam
from torchinfo import summary
import seaborn as sns
import matplotlib.pyplot as plt
from pyro.infer import MCMC, NUTS


# Define the basic FC model structure here, we will turn the model parameters into pyro data format in the model
# and guide function, so that the pyro could update the parameters.
class NN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output = self.fc1(x)
        output = F.relu(output)
        output = self.out(output)
        return output

# Load train dataset
train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist-data/', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),])),
        batch_size=128, shuffle=True)
# Load test dataset
test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist-data/', train=False, transform=transforms.Compose([transforms.ToTensor(),])
                       ),
        batch_size=1024, shuffle=True)
# create FC network
net = NN(28*28, 1024, 10)

log_softmax = nn.LogSoftmax(dim=1)
softplus=nn.Softplus()

# The model is used to simulate the posteriori of the observed data and conditional hidden variables
# To achieve this, we should give the prior of the hidden variables and chose a distribution for
# conditional probability. Here, the hidden variable is the model parameters w, we assume the p(w) follows the norm
# gaussian prior, note, this means that the w is a random variable, in pyro, we would represent it
# as pyro.sample('name',Norm(mean, variance)), it is not trainable!!!
# We then assume the distribution of the conditional probability is given by the FC neural network
# Where the model represents the P(D|w), we could calculate the posteriori by sampling p(w) from prior
# and then calculate P(D|w)*P(w)=P(D, w) by passing the x_data to model,
# but the FC neural network would only give a logits(distribution of the prediction)
# We simply select the probability that corresponding to the observed label as the final posteriori probability
def model(x_data, y_data):
    fc1w_prior = Normal(loc=torch.zeros_like(net.fc1.weight), scale=torch.ones_like(net.fc1.weight)).to_event(2)
    fc1b_prior = Normal(loc=torch.zeros_like(net.fc1.bias), scale=torch.ones_like(net.fc1.bias)).to_event(1)

    outw_prior = Normal(loc=torch.zeros_like(net.out.weight), scale=torch.ones_like(net.out.weight)).to_event(2)
    outb_prior = Normal(loc=torch.zeros_like(net.out.bias), scale=torch.ones_like(net.out.bias)).to_event(1)

    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior, 'out.weight': outw_prior, 'out.bias': outb_prior}
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", net, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()

    lhat = log_softmax(lifted_reg_model(x_data))
    pyro.deterministic("lhat", lhat)
    pyro.sample("obs", Categorical(logits=lhat).to_event(1), obs=y_data)
    return lhat


# The guide is used to simulate the posteriori distribution q(w), by reducing the KL divergence
# between qθ(w) and p(D, w), to reduce the kl divergence, for all the random variable that has the
# format pyro.sample('name',Norm(mean, variance)) in model function, we need to define a same
# random variable pyro.sample('name',Norm(mean, variance)) in guide function. The Kl divergence
# is calculated based on the probability difference of sampling the same value
# The trainable variable θ is represented as the pyro.param("name", value)
# here we assume the mean and variance of the w is learnable parameter θ
# by adjusting θ we reduce the difference between two distributions
def guide(x_data, y_data):

    # First layer weight distribution priors
    fc1w_mu = torch.randn_like(net.fc1.weight)
    fc1w_sigma = torch.randn_like(net.fc1.weight)
    fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
    fc1w_sigma_param = softplus(pyro.param("fc1w_sigma", fc1w_sigma))
    fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
    # First layer bias distribution priors
    fc1b_mu = torch.randn_like(net.fc1.bias)
    fc1b_sigma = torch.randn_like(net.fc1.bias)
    fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
    fc1b_sigma_param = softplus(pyro.param("fc1b_sigma", fc1b_sigma))
    fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)
    # Output layer weight distribution priors
    outw_mu = torch.randn_like(net.out.weight)
    outw_sigma = torch.randn_like(net.out.weight)
    outw_mu_param = pyro.param("outw_mu", outw_mu)
    outw_sigma_param = softplus(pyro.param("outw_sigma", outw_sigma))
    outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param)
    # Output layer bias distribution priors
    outb_mu = torch.randn_like(net.out.bias)
    outb_sigma = torch.randn_like(net.out.bias)
    outb_mu_param = pyro.param("outb_mu", outb_mu)
    outb_sigma_param = softplus(pyro.param("outb_sigma", outb_sigma))
    outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)
    priors = {'fc1.weight': fc1w_prior.to_event(2), 'fc1.bias': fc1b_prior.to_event(1),
              'out.weight': outw_prior.to_event(2), 'out.bias': outb_prior.to_event(1)}

    lifted_module = pyro.random_module("module", net, priors)

    return lifted_module()

# Use variance inference method to train a q(w) that simulates the p(D,w)
# need model, guide function for svi training, num_iterations is the number of training epochs
class SviTrainer():
    def __init__(self, model, guide, num_iterations=5):
        optim = ClippedAdam({"lr": 0.01})
        self.svi = SVI(model, guide, optim, loss=Trace_ELBO())
        self.num_iterations = num_iterations
        self.loss = 0
        self.model=model
        self.guide=guide

    def train(self):
        for j in range(self.num_iterations):
            loss = 0
            for batch_id, data in enumerate(train_loader):
                # calculate the loss and take a gradient step
                loss += self.svi.step(data[0].view(-1, 28 * 28), data[1])
            normalizer_train = len(train_loader.dataset)
            total_epoch_loss_train = loss / normalizer_train

            print("Epoch ", j, " Loss ", total_epoch_loss_train)

# including prediction and visualization for trained FC neural network
# sample_type could be either "svi" or "mcmc"
# if using svi, need to provide guide function in initialization
# if using mcmc, need to provide McmcTrainer.mcmc as mcmc_kernal in initialization
class Visualizer():
    def __init__(self, model, sample_type="svi", guide=None, mcmc_kernal=None, num_samples=100, num_class=10):
        self.loss = 0
        self.num_samples = num_samples
        self.num_class = num_class
        self.model = model
        self.guide = guide
        self.sample_type = sample_type
        self.mcmc_kernal = mcmc_kernal

    # We are actually showing two methods of sampling w sets(model parameter sets) here
    # we get predictions from all sampled models and using mean value as the final prediction
    def predict(self, x):
        if self.sample_type=="svi":
            sampled_models = [self.guide(None, None) for _ in range(self.num_samples)]
            yhats = torch.stack([F.log_softmax(model(x).data, 1) for model in sampled_models])
        elif self.sample_type=="mcmc":
            predictive = Predictive(model=self.model, posterior_samples=self.mcmc_kernal.get_samples())(x, None)
            yhats = predictive["lhat"]
        mean = torch.mean(yhats, 0)
        return np.argmax(mean.numpy(), axis=1)

    def directly_prediction(self):
        print('Prediction when network is forced to predict')
        correct = 0
        total = 0
        for j, data in enumerate(test_loader):
            images, labels = data
            predicted = self.predict(images.view(-1,28*28))
            total += labels.size(0)
            predicted=torch.tensor(predicted)
            correct += (predicted == labels).sum().item()

        print("accuracy: %d %%" % (100 * correct / total))
        print(f"correct: {correct}")
        print(f"total: {total}")

    # actually the same with predict(self, x) function
    # but return softmax logits instead of log softmax logits
    def give_uncertainities(self, x):
        if self.sample_type == "svi":
            sampled_models = [self.guide(None, None) for _ in range(self.num_samples)]
            yhats = [F.softmax(model(x.view(-1,28*28)).data, 1).detach().numpy() for model in sampled_models]
        elif self.sample_type == "mcmc":
            predictive = Predictive(model=self.model, posterior_samples=self.mcmc_kernal.get_samples())(x.view(-1,28*28), None)
            yhats = np.exp(predictive["lhat"])
        return np.asarray(yhats)
        #mean = torch.mean(torch.stack(yhats), 0)
        #return np.argmax(mean, axis=1)


    def imshow(self, img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        # plt.imshow(npimg,  cmap='gray')
        # fig.show(figsize=(1,1))

        fig, ax = plt.subplots(figsize=(1, 1))
        ax.imshow(npimg, cmap='gray', interpolation='nearest')
        plt.show()

    # we may choose not to predict for some images as we may have low confidence on them
    # this should depend on the mean and variance of sampled predictions (confidence interval)
    # You may change the decision rule
    def confidence(self, images, labels, threshold=0.6, plot=True):
        y = self.give_uncertainities(images)
        predicted_for_images = 0
        correct_predictions = 0
        total_images = len(labels)

        for i in range(len(labels)):
            if (plot):
                print("\nReal Lable: ", labels[i].item())
                fig, axs = plt.subplots(1, 10, sharey=True, figsize=(20, 2))

            all_digits_prob = []
            predict_stats = {}
            highted_something = False

            for j in range(self.num_class):

                highlight = False

                histo = []

                for z in range(y.shape[0]):
                    histo.append(y[z][i][j])

                marginal_site = pd.DataFrame(histo)
                describe = marginal_site.describe(percentiles=[.05, 0.25, 0.5, 0.75, 0.95]).transpose()
                predict_stats[j] = {}
                predict_stats[j]["percentiles"] = describe[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
                # print(f"Histogram for label {j}: {histo}")
                data=np.array(histo)
                np.round(data, decimals=5)

                confidence_level = 0.95
                mean = np.mean(data)
                std_error = stats.sem(data)
                margin_of_error = std_error * stats.t.ppf((1 + confidence_level) / 2, len(data) - 1)
                lower_bound = mean - margin_of_error
                upper_bound = mean + margin_of_error
                predict_stats[j]["confidence"]="95% confidence interval: ({}, {})".format(lower_bound, upper_bound)
                all_digits_prob.append(mean)
                if mean > threshold:
                    highlight=True

                if (plot):
                    N, bins, patches = axs[j].hist(histo, bins=8, color="lightgray", lw=0,
                                                   weights=np.ones(len(histo)) / len(histo), density=False, range=[0, 1])
                    axs[j].set_xlim([-0.01, 1.01])
                    axs[j].set_ylim([0, 1])
                    axs[j].set_xlabel("Model Output Score")
                    axs[j].set_ylabel("Sample Score Density")
                    axs[j].set_title(str(j) + " (" + str(round(mean, 2)) + ")")

                if (highlight):

                    highted_something = True

                    if (plot):

                        # We'll color code by height, but you could use any scalar
                        fracs = N / N.max()

                        # we need to normalize the data to 0..1 for the full range of the colormap
                        norm = colors.Normalize(fracs.min(), fracs.max())

                        # Now, we'll loop through our objects and set the color of each accordingly
                        for thisfrac, thispatch in zip(fracs, patches):
                            color = plt.cm.viridis(norm(thisfrac))
                            thispatch.set_facecolor(color)

            if (plot):
                plt.show()

            predicted = np.argmax(all_digits_prob)

            if (highted_something):
                predicted_for_images += 1
                if (labels[i].item() == predicted):
                    if (plot):
                        print("Correct")
                    correct_predictions += 1.0
                else:
                    if (plot):
                        print("Incorrect :()")
            else:
                if (plot):
                    print("Undecided.")

            if (plot):
                self.imshow(images[i].squeeze())

                for k, v in predict_stats.items():
                    print("\nlabel {}".format(k))
                    print(v["percentiles"])
                    print(v["confidence"])
        return (correct_predictions / predicted_for_images * 100, predicted_for_images/total_images * 100)

    def prediction_with_uncertainty(self, plot_uncertainty_range=False):
        if plot_uncertainty_range:
            percentages = []
            for uncertainty in np.arange(start=0, stop=1.0, step=0.05):
                count = 0
                for j, data in enumerate(test_loader):
                    images, labels = data
                    percentages.append((self.confidence(images, labels, threshold=uncertainty, plot=False), uncertainty))
                    break
            print(f"Average Correct Prediction %: {percentages}")
        else:
            for j, data in enumerate(test_loader):
                images, labels = data
                self.confidence(images, labels, threshold=0.6)
'''
learned_alpha = pyro.param("fc1.weight").item()
learned_beta = pyro.param("fc1.bias").item()
print("Learned alpha:", learned_alpha)
print("Learned beta:", learned_beta)
'''
def get_params(print_samples=False):
    for k, v in pyro.get_param_store().items():
        print(k, v)

    for j, data in enumerate(test_loader):
        images, labels = data
        predictive_svi = Predictive(model, guide=guide, num_samples=100)(images.view(-1,28*28), None)

        if print_samples:
            for k, v in predictive_svi.items():
                print(k, v.shape)
            obs_data=predictive_svi["obs"]
            unique_values, counts = torch.unique(obs_data[:, 1], return_counts=True)
            # Find the most frequent value(s)
            predicted, count = torch.mode(obs_data, dim=0)
            print(predicted)
            print((predicted == labels).sum().item())

    keys=["module$$$fc1.weight", "module$$$fc1.bias", "module$$$out.weight", "module$$$out.bias"]


    sns.histplot(predictive_svi["module$$$fc1.weight"][:, 0, 0], kde=True, color='skyblue',stat='probability')

    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution Plot')

    # Show the plot
    plt.show()


fc1w_mu=None
fc1w_sigma=None
fc1b_mu=None
fc1b_sigma=None
outw_mu = None
outw_sigma = None
outb_mu = None
outb_sigma = None
def posterior_model(x_data, y_data):

    fc1w_prior = Normal(loc=fc1w_mu, scale= softplus(fc1w_sigma)).to_event(2)
    fc1b_prior = Normal(loc=fc1b_mu, scale=softplus(fc1b_sigma)).to_event(1)

    outw_prior = Normal(loc=outw_mu, scale=softplus(outw_sigma)).to_event(2)
    outb_prior = Normal(loc= outb_mu, scale=softplus(outb_sigma)).to_event(1)

    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior, 'out.weight': outw_prior, 'out.bias': outb_prior}
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", net, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()

    lhat = log_softmax(lifted_reg_model(x_data))

    pyro.sample("obs", Categorical(logits=lhat).to_event(1), obs=y_data)


svi_trainer=SviTrainer(model=model, guide=guide, num_iterations=5)
svi_trainer.train()

svi_visualize=Visualizer(model=model, guide=guide)
# svi_visualize.directly_prediction()
svi_visualize.prediction_with_uncertainty()


fc1w_mu=pyro.get_param_store()['fc1w_mu']
fc1w_sigma=pyro.get_param_store()['fc1w_sigma']
fc1b_mu=pyro.get_param_store()['fc1b_mu']
fc1b_sigma=pyro.get_param_store()['fc1b_sigma']
outw_mu = pyro.get_param_store()['outw_mu']
outw_sigma = pyro.get_param_store()['outw_sigma']
outb_mu = pyro.get_param_store()['outb_mu']
outb_sigma = pyro.get_param_store()['outb_sigma']
pyro.get_param_store().clear()
