"""
"""
import matplotlib.pyplot as plt
import numpy as np


def main():
    cached_observation = np.loadtxt(".cache/observations.txt")
    data_observation = np.loadtxt("data/observations.txt")

    cached_latents = np.loadtxt(".cache/latents.txt")
    data_latents = np.loadtxt("data/latents.txt")

    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(cached_observation[:, 0])
    ax[0].plot(data_observation[:, 0])
    ax[1].plot(np.log(cached_latents[:, 0]))
    ax[1].plot(np.log(data_latents[:, 0]))
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
