"""Trial of the particle filter in TensorFlow Probability.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

import partical_xspec as px

import util

sns.set_style("whitegrid")


def plot_and_save_particle_distribution_with_latents(
        particles, latents_true=None, times=None,
        particle_quantiles=None):

    if latents_true.shape[-2] != particles.shape[-3]:
        raise ValueError(
            "`latents_true.shape[-2]` must be `particles.shape[-3]`. "
            f"{latents_true.shape[-2]} != {particles.shape[-3]}.")

    if times is None:
        times = np.arange(particles.shape[-3])

    fig, ax = plt.subplots(2, sharex=True)

    if latents_true is not None:
        ax[0].plot(times, latents_true[:, 0], color="k")
        ax[1].plot(times, latents_true[:, 1], color="k")

    y_centers = np.quantile(particles, 0.5, axis=-2)
    ax[0].plot(times, y_centers[:, 0], color="r")
    ax[1].plot(times, y_centers[:, 1], color="r")

    if particle_quantiles is not None:
        for quantile in particle_quantiles:
            errors_sigma = np.quantile(particles, quantile, axis=-2)
            for i in range(2):
                ax[i].fill_between(
                    times, *errors_sigma[..., i], alpha=0.20,
                    facecolor="none", color="r", edgecolor="none")

    ax[-1].set_xlabel("Time")
    ax[0].set_ylabel("powerlaw.PhoIndex")
    ax[1].set_ylabel("powerlaw.norm")
    fig.align_ylabels()
    plt.tight_layout()

    try:
        if sys.argv[1] == "test":
            savepath = util.join_and_create_directory(
                ".cache", "figs", "curve_particle_filtered.png")
            plt.savefig(savepath, dpi=150)
    except IndexError:
        pass

    plt.show()
    plt.close()


def set_particle_numbers():
    try:
        if sys.argv[1] == "test":
            num_particles = 100
    except IndexError:
        num_particles = 10000
    return num_particles


def main():

    order = 1
    xspec_param_size = 2

    px.xspec.set_energy(0.5, 10.0, 10)
    num_particles = set_particle_numbers()

    dtype = tf.float32
    blockwise_bijector = tfb.Blockwise(
        bijectors=[tfb.Chain([tfb.Scale(1.0), tfb.Exp()]),
                   tfb.Chain([tfb.Scale(10.), tfb.Exp()])]
    )

    observations = tf.convert_to_tensor(
        np.loadtxt(".cache/observations.txt"),
        dtype=dtype)

    def get_log_lik():

        def _get_log_lik(params):

            coefficients = tf.linalg.diag(
                tf.reshape(params, (order, xspec_param_size)))
            noise_scales = np.array([0.3, 0.3])
            transition_noise_cov = np.diag(
                tf.convert_to_tensor(noise_scales**2, dtype=dtype))
            var_trans = px.TransitionVectorAutoregressive(
                coefficients, transition_noise_cov, dtype=dtype)

            transition_fn = var_trans.transition_function

            ids_latent = var_trans.default_using_latent_indicies
            observation_fn = px.get_observaton_function_xspec_poisson(
                "powerlaw", xspec_param_size, num_particles,
                ids_latent, bijector=blockwise_bijector)

            initial_state_prior = tfd.MultivariateNormalDiag(
                loc=tf.repeat(0.1, order*xspec_param_size),
                scale_diag=tf.repeat(0.01, order*xspec_param_size))

            particles, _, _, log_lik = tfp.experimental.mcmc.particle_filter(
                observations,
                initial_state_prior,
                transition_fn,
                observation_fn,
                num_particles,
                parallel_iterations=1,
                seed=0
            )

            log_lik_mean = tf.reduce_mean(log_lik, axis=-1)
            return log_lik_mean

        return _get_log_lik

    tf.random.set_seed(123)

    # Allow external control of sampling to reduce test runtimes.
    num_results = 10
    num_results = int(num_results)

    num_burnin_steps = 1
    num_burnin_steps = int(num_burnin_steps)

    init_state = [
        tfd.MultivariateNormalDiag(
            scale_diag=tf.ones(xspec_param_size)).sample(seed=123)
    ]
    unconstrained_bijector = [
        tfb.Tanh(),
    ]
    log_lik_fn = get_log_lik()
    sampler = tfp.mcmc.TransformedTransitionKernel(
        tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=log_lik_fn,
            step_size=0.1),
        bijector=unconstrained_bijector)

    adaptive_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=sampler,
        num_adaptation_steps=int(0.8 * num_burnin_steps),
        target_accept_prob=0.75)

    import time

    # Speed up sampling by tracing with `tf.function`.
    # @tf.function(autograph=False, jit_compile=True)
    def do_sampling():
        return tfp.mcmc.sample_chain(
            current_state=init_state,
            kernel=adaptive_sampler,
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            trace_fn=None,
            seed=456)

    t0 = time.time()
    mcmc_samples_list = do_sampling()
    t1 = time.time()
    print("Inference ran in {:.2f}s.".format(t1-t0))

    print(mcmc_samples_list)

    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(mcmc_samples_list[0][:, 0])
    ax[1].plot(mcmc_samples_list[0][:, 1])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
