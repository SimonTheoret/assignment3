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
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sample = self.mean + self.std * torch.randn(self.mean.shape).to(device)
        else:
            sample = self.mean + self.std * noise
        return sample

    def kl(self):
        # Compute the KL-Divergence between the distribution with the standard normal N(0, I)
        # Return: Tensor of size (batch size,) containing the KL-Divergence for each element in the batch
        # from the wiki article: https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback%E2%80%93Leibler_divergence

        kl_div = None  # WRITE CODE HERE
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # var_inv = torch.pow(self.var, -1)
        dim = self.mean.shape[1]  # dimensions of a single example
        trace = torch.sum(self.var)
        prod = torch.matmul(
            self.mean, torch.eye(dim).to(device) @ torch.transpose(self.mean, -1, -2)
        ).squeeze()
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

    def nll(self, sample, dims=[1, 2, 3]):  # BUG: dim not OK
        # Computes the negative log likelihood of the sample under the given distribution
        # Return: Tensor of size (batch size,) containing the log-likelihood for each element in the batch
        # from the wiki article: https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Likelihood_function
        # shape var == shape sample = shape mean
        var = self.var
        var_inv = torch.pow(var, -1)
        diff = sample - self.mean
        print(dims)
        det = torch.prod(var,dim = dims[-1])
        print("passe 1")
        for dim in dims[:-1]:
            det *= torch.prod(var, dim = dim)
        print("passe 2")
        log_det = torch.log(det)
        print("passe 3")
        elem = (diff * var_inv)
        print(elem.shape)
        print(diff.shape)
        shape_pos =  len(diff.shape)-1
        dist = torch.matmul(elem , torch.transpose(diff, shape_pos-2, shape_pos-1 )) if len(dims)>=2 else torch.matmul(elem, diff.T)
        const = log(2 * pi)
        for i in range(1, len(sample.shape)):
            const*= sample.shape[i]
        print("passe 4")
        print(log_det.shape)
        print(dist.shape)
        negative_ll = 0.5 * torch.sum(log_det + dist + const, dim=dims)
        return negative_ll

    # negative_ll = None  # WRITE CODE HERE
    # log_det = torch.log(torch.det(self.var))
    # diff = sample - self.mean
    # dist = diff @ torch.inverse(self.var) @ diff
    # const = self.mean.shape[0]*self.mean.shape[1]*self.mean.shape[2]*log(2*pi)
    # negative_ll  = -0.5 *( log_det + dist + const )
    # return negative_ll

    # old:
    # batch_size x channel x image_dim (64 x 1 x 32^2)
    # var = torch.flatten(self.var, start_dim = 2)
    # var = torch.pow(var, -1)
    # mean = torch.flatten(self.mean, start_dim = 2)
    # sample = torch.flatten(sample, start_dim = 2)
    # # log_det = torch.log(torch.prod(var, dim = 2))
    # log_det = torch.sum(torch.log(var), dim = 2)
    # diff = sample-mean
    # prod = torch.bmm(diff*var, torch.transpose(diff, 1, 2))
    # const = mean.shape[2] * log(2*pi)
    # # print(log_det)
    # return -0.5 * (log_det + prod + const)

    # old:
    # prod_1 = (var_inv @ (sample - mean))
    # diff = sample-self.mean
    # prod = (sample - self.mean) @ prod_1 #BUG: mauvaise dim
    # negative_ll = 0.5 * (log_det + prod + const)
    # return negative_ll
    # old:
    # mean = torch.reshape(self.mean, (64, 1, 32**2))
    # var = torch.reshape(self.var, (64, 1, 32**2))
    # sample = torch.reshape(sample, (64, 1, 32**2))
    # val = torch.nn.functional.gaussian_nll_loss(input = mean, target = sample, var = var, full = True)
    # print(val)
    # return val

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
        return DiagonalGaussianDistribution(mean=mean, logvar=logvar)

    def decode(self, z):
        # Input:
        #   z: Tensor of shape (batch_size, z_dim)
        # Returns
        #   conditional distribution: The likelihood distribution p_\theta(x | z)

        # WRITE CODE HERE
        z = self.decoder(z)
        id = torch.ones_like(z)
        return DiagonalGaussianDistribution(mean=z, logvar=id)

    def sample(self, batch_size, noise=None):
        # Input:
        #   batch_size: The number of samples to generate
        # Returns:
        #   samples: Generated samples using the decoder
        #            Size: (batch_size, 3, 32, 32)
        # WRITE CODE HERE
        if noise == None:
            z = torch.randn(batch_size, 3, 32, 32).to(self.device)
        else:
            z = noise
        return self.decode(z).mode()
        # WRITE CODE HERE

    def log_likelihood(self, x, K=100):  # HACK: this might not work att all
        # BUG: If there is a bug, it's most likely in here
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
            z = posterior.sample()  # WRITE CODE HERE (sample from q_phi)
            recon = self.decode(
                z
            )  # WRITE CODE HERE (decode to conditional distribution)
            print("prio not ok")
            shape = list(range(len(z.shape)))[:-1]
            prio = -prior.nll(z, dims = shape)
            print("prio ok")
            posterio = posterior.nll(z, dims = shape)
            print("posterio ok")
            recon = -recon.nll(x, dims = shape)
            print("recon ok")
            log_likelihood[:, i] = (
                prio + posterio + recon
            )  # WRITE CODE HERE (log of the summation
            # terms in approximate log-likelihood, that is, log p_\theta(x, z_i) -
            # log q_\phi(z_i | x))

            del z, recon

        ll = (
            1 / K * torch.logsumexp(log_likelihood, dim=-1)
        )  # WRITE CODE HERE (compute the final log-likelihood using the log-sum-exp trick)
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
        post = self.encode(x)
        print("post ok")
        z = post.sample(noise)
        print("z ok")
        recon = self.decode(z)
        print("recon ok")
        mode = recon.mode()
        print("mode ok")
        nll = -self.log_likelihood(x)
        print("log likelihood ok")
        kl = post.kl()
        print("KL ok")
        return mode, nll, kl

        # old:
        # posterior = self.encode(x)  # WRITE CODE HERE
        # latent_z = posterior.sample(noise)  # WRITE CODE HERE (sample a z)
        # recon = self.decode(latent_z)  # WRITE CODE HERE (decode)
        # return recon.mode(), self.log_likelihood(x), posterior.kl()


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
    z = (
        z_1 * lengths + (1 - lengths) * z_2
    )  # WRITE CODE HERE (interpolate z_1 to z_2 with n_samples points)
    return model.decode(z).mode()
