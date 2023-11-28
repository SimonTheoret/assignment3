import torch

discriminator_optimizer = None    # WRITE CODE HERE
generator_optimizer = None        # WRITE CODE HERE

criterion = None    # WRITE CODE HERE

def discriminator_train(discriminator, generator, real_samples, fake_samples):
  # Takes as input real and fake samples and returns the loss for the discriminator
  # Inputs:
  #   real_samples: Input images of size (batch_size, 3, 32, 32)
  #   fake_samples: Input images of size (batch_size, 3, 32, 32)
  # Returns:
  #   loss: Discriminator loss

  ones = None   # WRITE CODE HERE (targets for real data)
  zeros = None  # WRITE CODE HERE (targets for fake data)

  real_output = None    # WRITE CODE HERE (output of discriminator on real data)
  fake_output = None    # WRITE CODE HERE (output of discriminator on fake data)

  loss = None           # WRITE CODE HERE (define the loss based on criterion and above variables)

  return loss

def generator_train(discriminator, generator, fake_samples):
  # Takes as input fake samples and returns the loss for the generator
  # Inputs:
  #   fake_samples: Input images of size (batch_size, 3, 32, 32)
  # Returns:
  #   loss: Generator loss

  ones = None   # WRITE CODE HERE (targets for fake data but for generator loop)

  output = None # WRITE CODE HERE (output of the discriminator on the fake data)

  loss = None   # WRITE CODE HERE (loss for the generator based on criterion and above variables)

  return loss

def sample(generator, num_samples, noise=None):
  # Takes as input the number of samples and returns that many generated samples
  # Inputs:
  #   num_samples: Scalar denoting the number of samples
  # Returns:
  #   samples: Samples generated; tensor of shape (num_samples, 3, 32, 32)

  with torch.no_grad():
    # WRITE CODE HERE (sample from p_z and then generate samples from it)
    pass


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

  return None