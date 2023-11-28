import torch
import torch.nn as nn
from math import pi, sqrt, log


class Encoder(nn.Module):
    def __init__(self, nc, nef, nz, isize, device):
        super(Encoder, self).__init__()

        # Device
        self.device = device

        # Encoder: (nc, isize, isize) -> (nef*8, isize//16, isize//16)
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, nef, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(nef),
            nn.Conv2d(nef, nef * 2, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(nef * 2),
            nn.Conv2d(nef * 2, nef * 4, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(nef * 4),
            nn.Conv2d(nef * 4, nef * 8, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(nef * 8),
        )

    def forward(self, inputs):
        batch_size = inputs.size(0)
        hidden = self.encoder(inputs)
        hidden = hidden.view(batch_size, -1)
        return hidden


class Decoder(nn.Module):
    def __init__(self, nc, ndf, nz, isize):
        super(Decoder, self).__init__()

        # Map the latent vector to the feature map space
        self.ndf = ndf
        self.out_size = isize // 16
        self.decoder_dense = nn.Sequential(
            nn.Linear(nz, ndf * 8 * self.out_size * self.out_size), nn.ReLU(True)



)

        self.decoder_conv = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 8, ndf * 4, 3, 1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 4, ndf * 2, 3, 1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 2, ndf, 3, 1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf, nc, 3, 1, padding=1),
        )

    def forward(self, input):
        batch_size = input.size(0)
        hidden = self.decoder_dense(input).view(
            batch_size, self.ndf * 8, self.out_size, self.out_size
        )
        output = self.decoder_conv(hidden)
        return output


class DiagonalGaussianDistribution(object):
    # Gaussian Distribution with diagonal covariance matrix
    def __init__(self, mean, logvar=None):
        super(DiagonalGaussianDistribution, self).__init__()
        # Parameters:
        # mean: A tensor representing the mean of the distribution
        # logvar: Optional tensor representing the log of the standard variance
        #         for each of the dimensions of the distribution

        self.mean = mean
        if logvar is None:
            logvar = torch.zeros_like(self.mean)
        self.logvar = torch.clamp(logvar, -30.0, 20.0)

        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self, noise=None):
        # Provide a reparameterized sample from the distribution
        # Return: Tensor of the same size as the mean
        sample = None  # WRITE CODE HERE
        if noise == None:
            sample = self.mean + self.std * torch.randn(self.mean.shape)
        else:
            sample = self.mean + self.std * noise
        return sample

    def kl(self):
        # Compute the KL-Divergence between the distribution with the standard normal N(0, I)
        # Return: Tensor of size (batch size,) containing the KL-Divergence for each element in the batch
        # from the wiki article: https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback%E2%80%93Leibler_divergence

        kl_div = None  # WRITE CODE HERE
        var_inv = torch.pow(self.var, -1)
        dim = self.mean.shape[1]  # dimensions of a single example
        trace = torch.sum(self.var)
        prod = torch.matmul(torch.flatten(self.mean), torch.flatten(self.mean))
        log_det = torch.log(torch.prod(self.var))
        kl_div = 0.5 * (trace + prod - dim - log_det)
        return kl_div

        # old:
        # cov = torch.flatten(self.var.pow_(-1))
        # print("cov:", cov)
        # inside = -0.5 * torch.dot(torch.dot((samples - self.mean) , cov) , (samples - self.mean))
        # print("inside:", inside)
        # samples_proba = (
        #     (2 * pi) ** (-dim / 2) @ torch.prod(cov) ** (-1 / 2) @ torch.exp(inside)
        # )  # TODO: changer puissance carrée -dim/2

        # print("samples_proba", samples_proba)

        # normal_inside = -0.5 * (samples).T @ (samples)
        # normal_proba = (2 * pi) ** (-dim / 2) @ torch.exp(normal_inside)

        # kl_div = nn.KLDivLoss(reduction="none")(samples_proba, normal_proba.log())
        # return kl_div

    def nll(self, sample, dims=[1, 2, 3]):
        # Computes the negative log likelihood of the sample under the given distribution
        # Return: Tensor of size (batch size,) containing the log-likelihood for each element in the batch
        # from the wiki article: https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Likelihood_function
        negative_ll = None  # WRITE CODE HERE
        log_det = torch.log(torch.prod(self.var))
        var_inv = torch.flatten(torch.pow(self.var, -1))
        prod = torch.matmul(sample - self.mean, var_inv * (sample - self.mean))
        const = self.mean.shape[1] * log(pi)
        negative_ll = 0.5 * (log_det + prod + const)
        return negative_ll

        # old:
        # dim = self.mean.shape
        # cov = torch.diag(self.var)
        # inside = -0.5 * (sample - self.mean).T @ torch.inversecov @ (sample - self.mean)
        # sample_proba = (
        #     (2 * pi) ** (-dim / 2) @ sqrt(torch.det(cov)) @ torch.exp(inside)
        # )  # TODO: changer puissance carrée -dim/2

        # return negative_ll

    def mode(self):
        # Returns the mode of the distribution
        # WRITE CODE HERE
        return self.mean


class VAE(nn.Module):
    def __init__(
        self,
        in_channels=3,
        decoder_features=32,
        encoder_features=32,
        z_dim=100,
        input_size=32,
        device=torch.device("cuda:0"),
    ):
        super(VAE, self).__init__()

        self.z_dim = z_dim
        self.in_channels = in_channels
        self.device = device

        # Encode the Input
        self.encoder = Encoder(
            nc=in_channels,
            nef=encoder_features,
            nz=z_dim,
            isize=input_size,
            device=device,
        )

        # Map the encoded feature map to the latent vector of mean, (log)variance
        out_size = input_size // 16
        self.mean = nn.Linear(encoder_features * 8 * out_size * out_size, z_dim)
        self.logvar = nn.Linear(encoder_features * 8 * out_size * out_size, z_dim)

        # Decode the Latent Representation
        self.decoder = Decoder(
            nc=in_channels, ndf=decoder_features, nz=z_dim, isize=input_size
        )

    def encode(self, x):
        # Input:
        #   x: Tensor of shape (batch_size, 3, 32, 32)
        # Returns:
        #   posterior: The posterior distribution q_\phi(z | x)
        #
        # WRITE CODE HERE
        x = self.encoder(x)
        mean = self.mean(x)
        logvar = self.logvar(x)
        return DiagonalGaussianDistribution(mean = mean, logvar = logvar)

    def decode(self, z):
        # Input:
        #   z: Tensor of shape (batch_size, z_dim)
        # Returns
        #   conditional distribution: The likelihood distribution p_\theta(x | z)

        # WRITE CODE HERE
        z = self.decoder(z)
        id = torch.ones_like(z)
        return DiagonalGaussianDistribution(mean = z, logvar = id)

    def sample(self, batch_size, noise=None):
        # Input:
        #   batch_size: The number of samples to generate
        # Returns:
        #   samples: Generated samples using the decoder
        #            Size: (batch_size, 3, 32, 32)
        # WRITE CODE HERE
        if noise == None:
            z = torch.randn(batch_size,3,32,32)
        else:
            z = noise
        return self.decode(z).mode()
        # WRITE CODE HERE

        pass

    def log_likelihood(self, x, K=100):
        # Approximate the log-likelihood of the data using Importance Sampling
        # Inputs:
        #   x: Data sample tensor of shape (batch_size, 3, 32, 32)
        #   K: Number of samples to use to approximate p_\theta(x)
        # Returns:
        #   ll: Log likelihood of the sample x in the VAE model using K samples
        #       Size: (batch_size,)
        posterior = self.encode(x)
        prior = DiagonalGaussianDistribution(torch.zeros_like(posterior.mean))

        log_likelihood = torch.zeros(x.shape[0], K).to(self.device)
        for i in range(K):
            z = None  # WRITE CODE HERE (sample from q_phi)
            recon = None  # WRITE CODE HERE (decode to conditional distribution)
            log_likelihood[
                :, i
            ] = None  # WRITE CODE HERE (log of the summation terms in approximate log-likelihood, that is, log p_\theta(x, z_i) - log q_\phi(z_i | x))
            del z, recon

        ll = None  # WRITE CODE HERE (compute the final log-likelihood using the log-sum-exp trick)
        return ll

    def forward(self, x, noise=None):
        # Input:
        #   x: Tensor of shape (batch_size, 3, 32, 32)
        # Returns:
        #   reconstruction: The mode of the distribution p_\theta(x | z) as a candidate reconstruction
        #                   Size: (batch_size, 3, 32, 32)
        #   Conditional Negative Log-Likelihood: The negative log-likelihood of the input x under the distribution p_\theta(x | z)
        #                                         Size: (batch_size,)
        #   KL: The KL Divergence between the variational approximate posterior with N(0, I)
        #       Size: (batch_size,)
        posterior = None  # WRITE CODE HERE
        latent_z = None  # WRITE CODE HERE (sample a z)
        recon = None  # WRITE CODE HERE (decode)

        return recon.mode(), recon.nll(x), posterior.kl()


def interpolate(model, z_1, z_2, n_samples):
    # Interpolate between z_1 and z_2 with n_samples number of points, with the first point being z_1 and last being z_2.
    # Inputs:
    #   z_1: The first point in the latent space
    #   z_2: The second point in the latent space
    #   n_samples: Number of points interpolated
    # Returns:
    #   sample: The mode of the distribution obtained by decoding each point in the latent space
    #           Should be of size (n_samples, 3, 32, 32)
    lengths = torch.linspace(0.0, 1.0, n_samples).unsqueeze(1).to(z_1.device)
    z = None  # WRITE CODE HERE (interpolate z_1 to z_2 with n_samples points)
    return model.decode(z).mode()
