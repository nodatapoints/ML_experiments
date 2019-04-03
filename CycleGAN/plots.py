__all__ = 'plot_stats', 'plot_samples'

import matplotlib.pyplot as plt


def plot_image(image_array):
    """Plots the given 2D Image array (shape=image_dims) as grayscale image"""
    plt.tight_layout()
    plt.imshow(image_array, cmap='Greys_r')
    plt.axis('off')


def plot_samples(history):
    """Saves every image array in the 2D array `history` as file under ./samples"""
    for epoch_i, samples in enumerate(history):
        for sample_i, sample in enumerate(samples):
            plot_image(sample)
            plt.savefig(f'samples/{epoch_i:03.0f}_{sample_i:02.0f}.png')


def plot_stats(stats):
    fig, (ax_stacked_loss, ax_inv_loss, ax_acc) = plt.subplots(3, 1)
    fig.set_size_inches(16, 8)

    ax_stacked_loss.set_title('stacked loss')
    ax_stacked_loss.plot(stats.fs_loss)
    ax_stacked_loss.plot(stats.gs_loss)
    ax_stacked_loss.legend(('f', 'g'))

    ax_inv_loss.set_title('cycle loss')
    ax_inv_loss.plot(stats.leftinv_loss)
    ax_inv_loss.plot(stats.rightinv_loss)
    ax_inv_loss.legend(('left', 'right'))

    ax_acc.set_title('discriminator accuracy')
    ax_acc.plot(stats.fd_acc)
    ax_acc.plot(stats.gd_acc)
    ax_acc.legend(('f', 'g'))

    fig.savefig('stats.png')
