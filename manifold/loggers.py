import abc
from typing import Any, Optional, Sequence, Union

import pandas
import torch

__all__ = (
    "logger",
    "get_logger",
    "Logger",
)


# functional interface


def logger(name: str, **kwargs) -> "Logger":
    """Maps a logger name to a `Logger`."""
    if name == "in-memory":
        return InMemoryLogger(**kwargs)
    if name == "dataframe":
        return DataFrameLogger(**kwargs)
    if name == "tensorboard":
        return TensorBoardLogger(**kwargs)
    raise ValueError(f"logger '{name}' is not supported")


get_logger = logger


# class interface


class Logger(abc.ABC):
    """Base class for all Loggers."""

    def __init__(self) -> None:
        pass

    def log(self, step: int, **kwargs) -> None:
        pass

    def snapshot(self, epoch: int, **kwargs) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class InMemoryLogger(Logger):
    """In-memory logger."""

    def __init__(self) -> None:
        super().__init__()
        self._log_buffer = {}
        self._snapshot_buffer = {}

    @property
    def log_buffer(self) -> dict:
        return self._log_buffer

    @property
    def snapshot_buffer(self) -> dict:
        return self._snapshot_buffer

    def log(self, step: int, **kwargs) -> None:
        self._log_buffer.setdefault(step, {})
        self._log_buffer[step]["step"] = step
        self._log_buffer[step].update(kwargs)

    def snapshot(self, epoch: int, **kwargs) -> None:
        self._snapshot_buffer.setdefault(epoch, {})
        self._snapshot_buffer[epoch]["epoch"] = epoch
        self._snapshot_buffer[epoch].update(kwargs)

    def flush(self) -> None:
        self._log_buffer.clear()
        self._snapshot_buffer.clear()

    def close(self) -> None:
        self.flush()
        del self._log_buffer
        del self._snapshot_buffer


class DataFrameLogger(InMemoryLogger):
    """DataFrame logger."""

    @property
    def log_buffer(self) -> pandas.DataFrame:
        return pandas.DataFrame.from_dict(self._log_buffer, orient="index")

    @property
    def snapshot_buffer(self) -> pandas.DataFrame:
        return pandas.DataFrame.from_dict(self._snapshot_buffer, orient="index")

    def loss(self, epoch: int) -> float:
        df = self.log_buffer
        return df.loc[df["epoch"] == epoch]["loss"].mean().item()


class TensorBoardLogger(Logger):
    """TensorBoard logger.

    Reference:
        [1] https://pytorch.org/docs/stable/tensorboard.html
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ModuleNotFoundError:
            raise RuntimeError(
                "'TensorBoard' is not installed, run `pip install tensorboard` to install"
            )
        self._writer = SummaryWriter(log_dir, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError("accessing private attribute is not allowed")
        return getattr(self._writer, name)

    def log(self, step: int, **kwargs) -> None:
        for key, value in kwargs.items():
            self._writer.add_scalar(
                key,
                value,
                global_step=step,
                new_style=True,
            )

    def snapshot(self, epoch: int, **kwargs) -> None:
        # TODO
        raise NotImplementedError

    def add_embedding(
        self, step: int, embedding: torch.Tensor, **kwargs
    ) -> None:
        self._writer.add_embedding(
            embedding,
            tag="embedding",
            global_step=step,
            **kwargs,
        )

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()
        del self._writer
