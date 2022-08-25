import os

import matplotlib.pyplot as plt
import numpy as np


def fig_size(width_pt, fraction=1, ratio=(5 ** .5 - 1) / 2, subplots=(1, 1)):
    """
    Returns the width and heights in inches for a matplotlib figure.

    Parameters
    ----------
    width_pt : float
        Document width in points, in latex can be determined with `\showthe\textwidth`.
    fraction : float, optional
        The fraction of the width with which the figure will occupy. Default 1.
    ratio : float, optional
        Ratio of the figure. Default is the golden ratio.
    subplots : tuple, optional
        The shape of subplots.

    Returns
    -------
    fig_width_in : float
        Width of the figure in inches.
    fig_height_in : float
        Height of the figure in inches.
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = ratio

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return fig_width_in, fig_height_in


def new_fig(width_pt=420, fraction=1, ratio=(5 ** .5 - 1) / 2, subplots=(1, 1)):
    """
    Creates new instance of a `matplotlib.pyplot.figure` by using the `fig_size` function.

    Parameters
    ----------
    width_pt : float, optional
        Document width in points, in latex can be determined with `\showthe\textwidth`.
        Default is 420.
    fraction : float, optional
        The fraction of the width with which the figure will occupy. Default 1.
    ratio : float, optional
        Ratio of the figure. Default is the golden ratio.
    subplots : tuple, optional
        The shape of subplots.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        The figure.
    """

    fig = plt.figure(figsize=fig_size(width_pt, fraction, ratio, subplots))
    return fig


def trajectories_plot(T, U, Y, Y_dmd=None):
    """
    Creates subplot of the input and output trajectory.

    Parameters
    ----------
    T : numpy.ndarray
        Time steps.
    U : numpy.ndarray
        Input Trajectory value at the time steps.
    Y : numpy.ndarray
        Output Trajectory value at the time steps.
    Y_dmd : numpy.ndarray, optional
        Approximated output trajectory value at the time steps. Default `None`.
    """
    fraction = 2
    fig = new_fig(fraction=fraction, subplots=(1, 2))

    # Input trajectory
    ax = fig.add_subplot(1, 2, 1)
    title = 'Training Input' if Y_dmd is None else 'Testing Input'
    ax.set_title(title)
    ax.set(xlim=[np.min(T), np.max(T)])
    ax.set(xlabel='Time (s)')
    ax.plot(T, U[0], label=f'$u$')

    # Output trajectory
    ax = fig.add_subplot(1, 2, 2)
    title = 'Testing Output' if Y_dmd is None else 'Testing Output'
    ax.set_title(title)
    ax.set(xlim=[np.min(T), np.max(T)])
    ax.set(xlabel='Time (s)')
    ax.plot(T, Y[0], label=f'$y$')

    if Y_dmd is not None:
        ax.plot(T, Y_dmd[0], ls='--', label=r'$\widetilde{y}$')

    ax.legend(loc='best')


def passivity_plot(T, dHdt, S):
    """
    Visualizes the dissipation inequality.

    Parameters
    ----------
    T : numpy.ndarray
        Time steps.
    dHdt : numpy.ndarray
        Change of the storage function.
    S : numpy.ndarray
        Supply rate.
    """
    fraction = 1

    fig = new_fig(fraction=fraction)
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlim=[np.min(T), np.max(T)])
    ax.set(xlabel='Time (s)')
    ax.plot(T, dHdt[0] - S[0], label=r'$\frac{\mathrm{d}}{\mathrm{d}t}H(x) - \langle y(t),u(t)\rangle$')
    ax.plot(T, np.zeros(T.shape), ls='--', c='black')
    ax.legend(loc='best')


def magnitude_plot(w, lti):
    """
    Creates magnitude bode plot of a lti system.

    Parameters
    ----------
    w : numpy.ndarray
        Frequencies.
    lti : pymor.models.iosys.LTIModel
        (Error) lti system.
    """
    fraction = 1

    fig = new_fig(fraction=fraction)
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlim=[np.min(w), np.max(w)])
    lti.transfer_function.mag_plot(w, ax=ax)
