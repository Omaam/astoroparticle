import tensorflow as tf
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd


def main():
    dtype = tf.float32

    order = 1
    latent_size = 2

    num_coefficients = order * latent_size
    constrained_bijector = tfb.Tanh()
    coefficients_flat = tfd.MultivariateNormalDiag(
        scale_diag=tf.ones(num_coefficients, dtype=dtype)).sample()
    coefficients_flat = constrained_bijector.forward(coefficients_flat)
    coefficients = tf.linalg.diag(
        tf.reshape(coefficients_flat, (order, latent_size)))
    print(coefficients)

    coefficients_flat = tfd.TransformedDistribution(
        distribution=tfd.MultivariateNormalDiag(
            scale_diag=tf.ones(num_coefficients, dtype=dtype)
        ),
        bijector=tfb.Tanh()
    ).sample()
    coefficients = tf.linalg.diag(
        tf.reshape(coefficients_flat, (order, latent_size)))
    print(coefficients)


if __name__ == "__main__":
    main()
