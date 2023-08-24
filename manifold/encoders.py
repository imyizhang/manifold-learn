from typing import Callable, Optional, Sequence

import torch

# functional interface


def encoder(
    encoder_: str,
    /,
    in_features: int,
    out_features: int,
    **kwargs,
) -> torch.nn.Module:
    """Maps an encoder name to a `torch.nn.Module."""
    if encoder_ == "mlp":
        return MLP(in_features, out_features, **kwargs)
    raise ValueError(f"encoder '{encoder_}' is not supported")


# class interface


class MLP(torch.nn.Sequential):
    """Multilayer perceptron (MLP) module.

    Args:
        in_features (int): Number of features of each input sample.
        out_features (int): Number of features of each output sample.
        hidden_features (Sequence[int]): Sequence of hidden feature dimensions.
        bias (bool): Whether to use bias in the linear layers. Default to True.
        normalization (Callable[..., torch.nn.Module], optional): Normalization layers that will be stacked on top of the linear layers if not None. Default to None.
        activation (Callable[..., torch.nn.Module], optional): Activations that will be stacked on top of the normalization layers if used, otherwise on top of the linear layers. Default: torch.nn.ReLU.
        dropout (float): Probability for the dropout layers. Default to 0.0.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Sequence[int],
        bias: bool = True,
        normalization: Optional[Callable[..., torch.nn.Module]] = None,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        dropout: float = 0.0,
    ) -> None:
        layers = []
        for _out_features in hidden_features:
            layers.append(
                torch.nn.Linear(
                    in_features,
                    _out_features,
                    bias=bias,
                )
            )
            if normalization is not None:
                layers.append(normalization(_out_features))
            layers.append(activation())
            layers.append(torch.nn.Dropout(p=dropout))
            in_features = _out_features
        layers.append(torch.nn.Linear(in_features, out_features, bias=bias))
        super().__init__(*layers)
