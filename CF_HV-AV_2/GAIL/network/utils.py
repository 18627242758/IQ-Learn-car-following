import math
import torch
from torch import nn


def build_mlp(input_dim, output_dim, hidden_units=[64, 64],
              hidden_activation=nn.Tanh(), output_activation=None):
    """
    Build a MLP network
    Parameters
    ----------
    input_dim: int
        dimension of the input of the neural network
    output_dim: int
        dimension of the output of the neural network
    hidden_units: tuple
        hidden units of the neural network
    hidden_activation: nn.Module
        activation function of the hidden layers
    init: bool
        whether to init the neural network to be orthogonal weighted
    output_activation: nn.Module
        activation function of the output layer
    gain: float
        gain for the init function
    Returns
    -------
    nn: nn.Module
        MLP net
    """
    layers = []
    units = input_dim

    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units

    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)

    return nn.Sequential(*layers)


def calculate_log_pi(log_stds, noises, actions):
    """
    Calculate log(\pi(a|s)) given log(std) of the distribution, noises, and actions to take
    Parameters
    ----------
    log_stds: torch.Tensor
        log(std) of the distribution
    noises: torch.Tensor
        noises added to the action
    actions: torch.Tensor
        actions to take
    Returns
    -------
    log_pi: torch.Tensor
        log(\pi(a|s))
    """
    gaussian_log_probs = (-0.5 * noises.pow(2) - log_stds).sum(
        dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)

    return gaussian_log_probs - torch.log(
        1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)


def reparameterize(means, log_stds):
    """
    Get action and its log_pi according to mean and log_std
    Parameters
    ----------
    means: torch.Tensor
        mean value of the action
    log_stds: torch.Tensor
        log(std) of the action
    Returns
    -------
    actions: torch.Tensor
        actions to take
    log_pi: torch.Tensor
        log_pi of the actions
    """
    noises = torch.randn_like(means)
    us = means + noises * log_stds.exp()
    actions = torch.tanh(us)
    return actions, calculate_log_pi(log_stds, noises, actions)


def atanh(x):
    """
    Return atanh of the input. Modified torch.atanh in case the output is nan.
    Parameters
    ----------
    x: torch.Tensor
        input
    Returns
    -------
    y: torch.Tensor
        atanh(x)
    """
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))


def evaluate_lop_pi(means, log_stds, actions):
    """
    Evaluate the log(\pi(a|s)) of the given action
    Parameters
    ----------
    means: torch.Tensor
        mean value of the action distribution
    log_stds: torch.Tensor
        log(std) of the action distribution
    actions: torch.Tensor
        actions taken
    Returns
    -------
    log_pi: : torch.Tensor
        log(\pi(a|s))
    """
    noises = (atanh(actions) - means) / (log_stds.exp() + 1e-8)

    return calculate_log_pi(log_stds, noises, actions)
