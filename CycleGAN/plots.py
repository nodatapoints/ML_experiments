__all__ = 'plot_stats', 'plot_samples'

import matplotlib.pyplot as plt


def plot_image(image_array):
    """Plots the given 2D Image array (shape=image_dims) as grayscale image"""
    plt.tight_layout()
    plt.imshow(image_array, cmap='Greys_r')
    plt.axis('off')


def plot_samples(samples):
    for gen, (sample, name_fmt) in enumerate(samples):
        for i, image in enumerate(sample):
            plot_image(image)
            plt.savefig(name_fmt.format(gen=gen, i=i))


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
