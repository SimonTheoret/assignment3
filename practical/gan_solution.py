import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
# Training Hyperparameters
batch_size = 64  # Batch Size
z_dim = 32  # Latent Dimensionality
gen_lr = 1e-4  # Learning Rate for the Generator
disc_lr = 1e-4  # Learning Rate for the Discriminator/
# Define Dataset Statistics
image_size = 32
input_channels = 1


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, z_dim, channels, generator_features=32):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, generator_features * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(generator_features * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                generator_features * 4, generator_features * 2, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(generator_features * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                generator_features * 2, generator_features * 1, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(generator_features * 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(generator_features * 1, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

        self.apply(weights_init)

    def forward(self, input):
        return self.model(input)


class Discriminator(nn.Module):
    def __init__(self, channels, discriminator_features=32):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, discriminator_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                discriminator_features, discriminator_features * 2, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(discriminator_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                discriminator_features * 2,
                discriminator_features * 4,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(discriminator_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(discriminator_features * 4, 1, 4, 1, 0, bias=False),
        )

        self.apply(weights_init)

    def forward(self, input):
        return self.model(input)


generator = Generator(z_dim, input_channels).to(device)
discriminator = Discriminator(input_channels).to(device)

discriminator_optimizer = torch.optim.Adam(
    discriminator.parameters(), lr=disc_lr, betas=(0.5, 0.999)
)  # WRITE CODE HERE
generator_optimizer = torch.optim.Adam(
    generator.parameters(), lr=gen_lr, betas=(0.5, 0.999)
)  # WRITE CODE HERE

criterion = torch.nn.BCEWithLogitsLoss()  # WRITE CODE HERE


def discriminator_train(discriminator, generator, real_samples, fake_samples):
    # Takes as input real and fake samples and returns the loss for the discriminator
    # Inputs:
    #   real_samples: Input images of size (batch_size, 3, 32, 32)
    #   fake_samples: Input images of size (batch_size, 3, 32, 32)
    # Returns:
    #   loss: Discriminator loss

    real_output = discriminator(
        real_samples
    )  # WRITE CODE HERE (output of discriminator on real data)
    fake_output = discriminator(
        fake_samples
    )  # WRITE CODE HERE (output of discriminator on fake data)

    ones = torch.ones_like(real_output)  # WRITE CODE HERE (targets for real data)
    zeros = 1 - torch.ones_like(fake_output)  # WRITE CODE HERE (targets for fake data)

    loss = criterion(real_output, ones) + criterion(
        fake_output, zeros
    )  # WRITE CODE HERE (define the loss based on criterion and above variables)

    return loss


def generator_train(discriminator, generator, fake_samples):
    # Takes as input fake samples and returns the loss for the generator
    # Inputs:
    #   fake_samples: Input images of size (batch_size, 3, 32, 32)
    # Returns:
    #   loss: Generator loss

    output = discriminator(
        fake_samples
    )  # WRITE CODE HERE (output of the discriminator on the fake data)

    ones = torch.ones_like(
        output
    )  # WRITE CODE HERE (targets for fake data but for generator loop)

    loss = criterion(
        output, ones
    )  # WRITE CODE HERE (loss for the generator based on criterion and above variables)

    return loss


def sample(generator, num_samples, noise=None):
    # Takes as input the number of samples and returns that many generated samples
    # Inputs:
    #   num_samples: Scalar denoting the number of samples
    # Returns:
    #   samples: Samples generated; tensor of shape (num_samples, 3, 32, 32)

    with torch.no_grad():
        # WRITE CODE HERE (sample from p_z and then generate samples from it)
        pixel_space = torch.randn(num_samples, z_dim, 1, 1, device = device) if noise ==None else noise
        generated = generator(pixel_space)
        return generated


def interpolate(generator, z_1, z_2, n_samples):
    # Interpolate between z_1 and z_2 with n_samples number of points, with the first point being z_1 and last being z_2.
    # Inputs:
    #   z_1: The first point in the latent space
    #   z_2: The second point in the latent space
    #   n_samples: Number of points interpolated
    # Returns:
    #   sample: A sample from the generator obtained from each point in the latent space
    #           Should be of size (n_samples, 3, 32, 32)

    # WRITE CODE HERE (interpolate z_1 to z_2 with n_samples points and then)
    # WRITE CODE HERE (    generate samples from the respective latents     )
    inter = torch.zeros((n_samples, *z_1.shape))
    lengths = torch.linspace(0.0, 1.0, n_samples).to(z_1.device) # (n_samples)
    for i in range(n_samples):
        inter[i] = z_1*lengths[i]  + (1-lengths[i])*z_2
    # z = (
    #     z_1 * lengths + (1 - lengths) * z_2
    # )  # WRITE CODE HERE (interpolate z_1 to z_2 with n_samples points)
    # z = torch.tranpose(z, 2,)
    # print(z.shape)
    out = generator(inter)
    return out
