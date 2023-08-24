import os
from typing import Callable, Generic, Optional, Tuple, TypeVar, Union

import numpy

# functional interface

registry: dict = {}


def register(func):
    name = func.__name__
    while name.startswith("_"):
        name = name[1:]
    if name in registry:
        raise ValueError(f"dataset '{name}' is already registered")
    registry[name] = func
    return func


def download():
    """Downloads dataset."""
    # TODO
    raise NotImplementedError


def extract():
    """Extracts dataset."""
    # TODO
    raise NotImplementedError


def save(file, *args, **kwargs):
    """Saves dataset into a single file in compressed .npz format."""
    return numpy.savez_compressed(file, *args, **kwargs)


def load(file):
    """Loads dataset from .npz file."""
    return numpy.load(file, allow_pickle=True)


def to_random_state(
    seed: Optional[Union[int, numpy.random.RandomState]] = None,
) -> numpy.random.RandomState:
    """Returns a generator for random sampling."""
    if seed is None:
        return numpy.random.mtrand._rand
    if isinstance(seed, int):
        return numpy.random.RandomState(seed)
    if isinstance(seed, numpy.random.RandomState):
        return seed
    raise ValueError(
        f"'{seed}' is not supported to seed 'numpy.random.RandomState'"
    )


@register
def heart(
    loc: Tuple[float, float] = (0.0, 0.0),
    scale: float = 1.0,
    num_samples: int = 1000,
    random: bool = False,
    noise: Optional[float] = None,
    random_state: Optional[Union[int, numpy.random.RandomState]] = None,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Generates a 2D heart dataset."""
    generator = to_random_state(random_state)
    # 1D uniformly distributed
    if random:
        theta = generator.uniform(0, 2 * numpy.pi, num_samples)
    # 1D evenly distributed
    else:
        theta = numpy.linspace(0, 2 * numpy.pi, num_samples, endpoint=False)
    x, y = loc
    x += scale * numpy.sin(theta) ** 3
    y += (
        scale
        * (
            13 * numpy.cos(theta)
            - 5 * numpy.cos(2 * theta)
            - 2 * numpy.cos(3 * theta)
            - numpy.cos(4 * theta)
        )
        / 16
    )
    X = numpy.stack((x, y), axis=-1)
    if noise is not None and noise != 0:
        X += generator.normal(0, noise, X.shape)
    return X, theta


@register
def circle(
    loc: Tuple[float, float] = (0.0, 0.0),
    scale: float = 1.0,
    num_samples: int = 1000,
    random: bool = False,
    noise: Optional[float] = None,
    random_state: Optional[Union[int, numpy.random.RandomState]] = None,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Generates a 2D circle dataset."""
    generator = to_random_state(random_state)
    # 1D uniformly distributed
    if random:
        theta = generator.uniform(0, 2 * numpy.pi, num_samples)
    # 1D evenly distributed
    else:
        theta = numpy.linspace(0, 2 * numpy.pi, num_samples, endpoint=False)
    x, y = loc
    x += scale * numpy.cos(theta)
    y += scale * numpy.sin(theta)
    X = numpy.stack((x, y), axis=-1)
    if noise is not None and noise != 0:
        X += generator.normal(0, noise, X.shape)
    return X, theta


@register
def curve(
    low: float = 0.0,
    high: float = 1.0,
    num_samples: int = 1000,
    func: Optional[Callable] = None,
    random: bool = False,
    noise: Optional[float] = None,
    random_state: Optional[Union[int, numpy.random.RandomState]] = None,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Generates a 2D polynomial curve dataset."""
    generator = to_random_state(random_state)
    # 1D uniformly distributed
    if random:
        x = generator.uniform(low, high, num_samples)
    # 1D evenly distributed
    else:
        x = numpy.linspace(low, high, num_samples, endpoint=True)
    # polynomial function
    if func is None:
        func = (
            lambda x: 100
            * (x - 0.1)
            * (x - 0.2)
            * (x - 0.3)
            * (x - 0.8)
            * (x - 0.9)
        )
    y = func(x)
    X = numpy.stack((x, y), axis=-1)
    if noise is not None and noise != 0:
        X += generator.normal(0, noise, X.shape)
    return X, x


@register
def square(
    low: float = 0.0,
    high: float = 1.0,
    num_samples: int = 1000,
    random: bool = False,
    random_state: Optional[Union[int, numpy.random.RandomState]] = None,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Generates a 2D square dataset."""
    generator = to_random_state(random_state)
    # 2D uniformly distributed
    if random:
        return generator.uniform(low, high, (num_samples, 2)), None
    # 2D evenly distributed, a 2D square of grids
    size = int(numpy.sqrt(num_samples))
    x = numpy.linspace(low, high, size, endpoint=True)
    xx, yy = numpy.meshgrid(x, x)
    X = numpy.stack((xx, yy), axis=-1).reshape(-1, 2)
    return X, X[:, 0]


@register
def cluster(
    loc: Tuple[float, float] = (0.0, 0.0),
    scale: float = 1.0,
    num_samples: int = 1000,
    random_state: Optional[Union[int, numpy.random.RandomState]] = None,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Generates a 2D gaussian cluster dataset."""
    generator = to_random_state(random_state)
    mean = numpy.array(loc)
    cov = numpy.eye(2) * scale
    return generator.multivariate_normal(mean, cov, num_samples), None


@register
def clusters(
    loc: Tuple[float, float] = (0.0, 0.0),
    scale: float = 1.0,
    num_samples: int = 1000,
    num_clusters: int = 3,
    overlapping_factor: float = 3.0,
    increasing_distance: bool = False,
    increasing_variance: bool = False,
    increasing_sampling_rate: bool = False,
    random_state: Optional[Union[int, numpy.random.RandomState]] = None,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Generates a dataset with multiple 2D gaussian clusters."""
    generator = to_random_state(random_state)
    if scale < 1:
        distance = overlapping_factor * 2 * 1
    else:
        distance = overlapping_factor * 2 * scale
    mean = numpy.array(loc)
    cov = numpy.eye(2) * scale
    X = []
    labels = []
    for i in range(num_clusters):
        if increasing_distance and i > 1:
            distance *= 1.2 ** (i - 1)
        mean[0] = i * distance
        # recommend to set scale to 0.01
        if increasing_variance and i > 0:
            cov *= 10
        # recommend to set num_samples to 100
        if increasing_sampling_rate and i > 0:
            num_samples *= 10
        X.append(generator.multivariate_normal(mean, cov, num_samples))
        labels.append(i * numpy.ones(num_samples, dtype=numpy.int64))
    return numpy.vstack(X), numpy.concatenate(labels, axis=None)


@register
def s_curve(
    loc: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: float = 1.0,
    num_samples: int = 10000,
    random: bool = False,
    hole: bool = False,
    noise: Optional[float] = None,
    random_state: Optional[Union[int, numpy.random.RandomState]] = None,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Generates a 3D S-curve dataset."""
    generator = to_random_state(random_state)
    # compensate for hole since around 95% of expected samples would be drawn
    if hole:
        expected_num_samples = num_samples
        num_samples = int(num_samples / 0.94)
    # 1D uniformly distributed
    if random:
        theta = generator.uniform(0, 2 * numpy.pi, num_samples)
    # 1D evenly distributed
    else:
        theta = numpy.linspace(0, 2 * numpy.pi, num_samples, endpoint=False)
    t = 1.5 * (theta - numpy.pi)
    x, y, z = loc
    x += scale * numpy.sin(t)
    y += scale * generator.uniform(-1, 1, num_samples)
    z += scale * numpy.sign(t) * (numpy.cos(t) - 1)
    X = numpy.vstack((x, y, z)).T
    if noise is not None and noise != 0:
        X += generator.normal(0, noise, X.shape)
    if hole:
        anchor = numpy.array(loc) + scale * numpy.array([0, 0, 0])
        indices = numpy.sum(numpy.square(X - anchor), axis=1) > scale**2 * 0.3
        X, theta = X[indices], theta[indices]
        if len(X) > expected_num_samples:
            X, theta = X[:expected_num_samples], theta[:expected_num_samples]
    return X, theta


@register
def swiss_roll(
    loc: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: float = 1.0,
    num_samples: int = 10000,
    random: bool = False,
    hole: bool = False,
    noise: Optional[float] = None,
    random_state: Optional[Union[int, numpy.random.RandomState]] = None,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Generates a 3D Swiss roll dataset."""
    generator = to_random_state(random_state)
    # compensate for hole since around 95% of expected samples would be drawn
    if hole:
        expected_num_samples = num_samples
        num_samples = int(num_samples / 0.94)
    # 1D uniformly distributed
    if random:
        theta = generator.uniform(0, 2 * numpy.pi, num_samples)
    # 1D evenly distributed
    else:
        theta = numpy.linspace(0, 2 * numpy.pi, num_samples, endpoint=False)
    t = 1.5 * (theta + numpy.pi)
    x, y, z = loc
    x += scale * t * numpy.cos(t) / 10.5
    y += scale * generator.uniform(-1, 1, num_samples)
    z += scale * t * numpy.sin(t) / 10.5
    X = numpy.vstack((x, y, z)).T
    if hole:
        pass
    if noise is not None and noise != 0:
        X += generator.normal(0, noise, X.shape)
    if hole:
        anchor = numpy.array(loc) + scale * numpy.array([-1, 0, 0])
        indices = numpy.sum(numpy.square(X - anchor), axis=1) > scale**2 * 0.3
        X, theta = X[indices], theta[indices]
        if len(X) > expected_num_samples:
            X, theta = X[:expected_num_samples], theta[:expected_num_samples]
    return X, theta


def _dataset(
    file: str,
    root: str = "./",
    download: bool = False,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    if download:
        raise NotImplementedError
    file = os.path.join(root, file)
    # print(f"loading {file}")
    data = load(file)
    return data["X"], data["y"]


@register
def mammoth(
    root: str = "./",
    download: bool = False,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Returns Mammoth dataset."""
    if download:
        raise NotImplementedError
    file = os.path.join(root, "data/mammoth/mammoth.npz")
    # fixme: replace with loggings
    print(f"loading {file}")
    data = load(file)
    return data["X"], data["y"]


@register
def t_rex(
    root: str = "./",
    download: bool = False,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Returns T-Rex dataset."""
    if download:
        raise NotImplementedError
    file = os.path.join(root, "data/t-rex/t-rex.npz")
    # fixme: replace with loggings
    print(f"loading {file}")
    data = load(file)
    return data["X"], data["y"]


@register
def mnist(
    root: str = "./",
    download: bool = False,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Returns MNIST dataset."""
    if download:
        raise NotImplementedError
    file = os.path.join(root, "data/mnist/mnist.npz")
    # fixme: replace with loggings
    print(f"loading {file}")
    data = load(file)
    return data["X"], data["y"]


@register
def kuzushiji_mnist(
    root: str = "./",
    download: bool = False,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Returns Kuzushiji-MNIST dataset."""
    if download:
        raise NotImplementedError
    file = os.path.join(root, "data/kuzushiji-mnist/kmnist.npz")
    # fixme: replace with loggings
    print(f"loading {file}")
    data = load(file)
    return data["X"], data["y"]


@register
def fashion_mnist(
    root: str = "./",
    download: bool = False,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Returns Fashion-MNIST dataset."""
    if download:
        raise NotImplementedError
    file = os.path.join(root, "data/fashion-mnist/fmnist.npz")
    # fixme: replace with loggings
    print(f"loading {file}")
    data = load(file)
    return data["X"], data["y"]


@register
def cifar10(
    root: str = "./",
    download: bool = False,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Returns CIFAR-10 dataset."""
    if download:
        raise NotImplementedError
    file = os.path.join(root, "data/cifar-10/cifar-10.npz")
    # fixme: replace with loggings
    print(f"loading {file}")
    data = load(file)
    return data["X"], data["y"]


@register
def cifar100(
    root: str = "./",
    download: bool = False,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Returns CIFAR-100 dataset."""
    if download:
        raise NotImplementedError
    file = os.path.join(root, "data/cifar-100/cifar-100.npz")
    # fixme: replace with loggings
    print(f"loading {file}")
    data = load(file)
    return data["X"], data["y"]


@register
def coil20(
    root: str = "./",
    download: bool = False,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Returns COIL-20 dataset."""
    if download:
        raise NotImplementedError
    file = os.path.join(root, "data/coil-20/coil-20.npz")
    # fixme: replace with loggings
    print(f"loading {file}")
    data = load(file)
    return data["X"], data["y"]


@register
def coil100(
    root: str = "./",
    download: bool = False,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Returns COIL-100 dataset."""
    if download:
        raise NotImplementedError
    file = os.path.join(root, "data/coil-100/coil-100.npz")
    # fixme: replace with loggings
    print(f"loading {file}")
    data = load(file)
    return data["X"], data["y"]


@register
def tess(
    root: str = "./",
    download: bool = False,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Returns Tess dataset from FR-FCM-ZZ36."""
    if download:
        raise NotImplementedError
    file = os.path.join(root, "data/fr-fcm-zz36/tess.npz")
    # fixme: replace with loggings
    print(f"loading {file}")
    data = load(file)
    return data["X"], data["y"]


@register
def anna(
    root: str = "./",
    download: bool = False,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Returns Anna dataset from FR-FCM-ZZSC."""
    if download:
        raise NotImplementedError
    file = os.path.join(root, "data/fr-fcm-zzsc/anna.npz")
    # fixme: replace with loggings
    print(f"loading {file}")
    data = load(file)
    return data["X"], data["y"]


@register
def levine13dim(
    root: str = "./",
    download: bool = False,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Returns Levine_13dim dataset from FR-FCM-ZZPH."""
    if download:
        raise NotImplementedError
    file = os.path.join(root, "data/fr-fcm-zzph/levine_13dim.npz")
    # fixme: replace with loggings
    print(f"loading {file}")
    data = load(file)
    return data["X"], data["y"]


@register
def levine32dim(
    root: str = "./",
    download: bool = False,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Returns Levine_32dim dataset from FR-FCM-ZZPH."""
    if download:
        raise NotImplementedError
    file = os.path.join(root, "data/fr-fcm-zzph/levine_32dim.npz")
    # fixme: replace with loggings
    print(f"loading {file}")
    data = load(file)
    return data["X"], data["y"]


@register
def samusik01(
    root: str = "./",
    download: bool = False,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Returns Samusik_01 dataset from FR-FCM-ZZPH."""
    if download:
        raise NotImplementedError
    file = os.path.join(root, "data/fr-fcm-zzph/samusik_01.npz")
    # fixme: replace with loggings
    print(f"loading {file}")
    data = load(file)
    return data["X"], data["y"]


@register
def samusik(
    root: str = "./",
    download: bool = False,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Returns Samusik_all dataset from FR-FCM-ZZPH."""
    if download:
        raise NotImplementedError
    file = os.path.join(root, "data/fr-fcm-zzph/samusik_all.npz")
    # fixme: replace with loggings
    print(f"loading {file}")
    data = load(file)
    return data["X"], data["y"]


@register
def nilsson(
    root: str = "./",
    download: bool = False,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Returns Nilsson_rare dataset from FR-FCM-ZZPH."""
    if download:
        raise NotImplementedError
    file = os.path.join(root, "data/fr-fcm-zzph/nilsson_rare.npz")
    # fixme: replace with loggings
    print(f"loading {file}")
    data = load(file)
    return data["X"], data["y"]


@register
def mosmann(
    root: str = "./",
    download: bool = False,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Returns Mosmann_rare dataset from FR-FCM-ZZPH."""
    if download:
        raise NotImplementedError
    file = os.path.join(root, "data/fr-fcm-zzph/mosmann_rare.npz")
    # fixme: replace with loggings
    print(f"loading {file}")
    data = load(file)
    return data["X"], data["y"]


# class interface


T_co = TypeVar("T_co", covariant=True)


class Dataset(Generic[T_co]):
    """Base class for all Datasets."""

    def __init__(self) -> None:
        pass

    def __call__(self) -> T_co:
        raise NotImplementedError


class Heart(Dataset[Tuple[numpy.ndarray, numpy.ndarray]]):
    """2D heart dataset."""

    size: Optional[int] = None

    dimensions: int = 2

    labeled: bool = False

    def __call__(self, *args, **kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]:
        return heart(*args, **kwargs)


class Circle(Heart):
    """2D circle dataset.

    References:
        [1] Andy Coenen and Adam Pearce. "Understanding UMAP", 2019. https://pair-code.github.io/understanding-umap/
        [2] Wattenberg, et al., "How to Use t-SNE Effectively", Distill, 2016. https://distill.pub/2016/misread-tsne/
    """

    def __call__(self, *args, **kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]:
        return circle(*args, **kwargs)


class Square(Heart):
    """2D square dataset.

    References:
        [1] Andy Coenen and Adam Pearce. "Understanding UMAP", 2019. https://pair-code.github.io/understanding-umap/
        [2] Wattenberg, et al., "How to Use t-SNE Effectively", Distill, 2016. https://distill.pub/2016/misread-tsne/
    """

    def __call__(self, *args, **kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]:
        return square(*args, **kwargs)


class Cluster(Heart):
    """2D gaussian cluster dataset.

    References:
        [1] Andy Coenen and Adam Pearce. "Understanding UMAP", 2019. https://pair-code.github.io/understanding-umap/
        [2] Wattenberg, et al., "How to Use t-SNE Effectively", Distill, 2016. https://distill.pub/2016/misread-tsne/
    """

    def __call__(self, *args, **kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]:
        return cluster(*args, **kwargs)


class Clusters(Heart):
    """Dataset with multiple 2D gaussian clusters.

    References:
        [1] Andy Coenen and Adam Pearce. "Understanding UMAP", 2019. https://pair-code.github.io/understanding-umap/
        [2] Wattenberg, et al., "How to Use t-SNE Effectively", Distill, 2016. https://distill.pub/2016/misread-tsne/
    """

    def __call__(self, *args, **kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]:
        return clusters(*args, **kwargs)


class SCurve(Heart):
    """3D S-curve dataset.

    References:
        [1] https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_s_curve.html
    """

    dimensions: int = 3

    def __call__(self, *args, **kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]:
        return s_curve(*args, **kwargs)


class SwissRoll(SCurve):
    """Swiss roll dataset.

    References:
        [1] https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_swiss_roll.html
        [2] S. Marsland, “Machine Learning: An Algorithmic Perspective”, 2nd edition, Chapter 6, 2014. https://homepages.ecs.vuw.ac.nz/~marslast/Code/Ch6/lle.py
    """

    def __call__(self, *args, **kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]:
        return swiss_roll(*args, **kwargs)


class Mammoth(Dataset[Tuple[numpy.ndarray, numpy.ndarray]]):
    """3D mammoth (50K) dataset.

    References:
        [1] Andy Coenen and Adam Pearce. "Understanding UMAP", 2019. https://pair-code.github.io/understanding-umap/
        [2] Max Noichl. "Examples for UMAP reduction using 3D models of prehistoric animals", 2020. https://homepage.univie.ac.at/maximilian.noichl/post/mammoth/
        [3] The Smithsonian Institute. "Mammuthus primigenius (Blumbach)", 2021. https://3d.si.edu/object/3d/mammuthus-primigenius-blumbach:341c96cd-f967-4540-8ed1-d3fc56d31f12
    """

    size: int = 50000

    dimensions: int = 3

    labeled: bool = False

    url: str = "https://github.com/MNoichl/UMAP-examples-mammoth-/blob/master/mammoth_a.csv"

    def __call__(self, *args, **kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]:
        return mammoth(*args, **kwargs)


class TRex(Mammoth):
    """3D T-rex (50K) dataset.

    References:
        [1] Max Noichl. "Examples for UMAP reduction using 3D models of prehistoric animals", 2020. https://homepage.univie.ac.at/maximilian.noichl/post/mammoth/
        [2] The Smithsonian Institute. "Tyrannosaurus rex Osborn, 1905", 2022. https://3d.si.edu/object/nmnhpaleobiology_10250729
    """

    url: str = "https://github.com/MNoichl/UMAP-examples-mammoth-/blob/master/rexy_a.csv"

    def __call__(self, *args, **kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]:
        return t_rex(*args, **kwargs)


class MNIST(Dataset[Tuple[numpy.ndarray, numpy.ndarray]]):
    """MNIST dataset.

    References:
        [1] http://yann.lecun.com/exdb/mnist/
    """

    size: int = 70000

    dimensions: int = 28 * 28

    labeled: bool = True

    classes: tuple = (
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    )  # 10

    mirrors: list = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
    ]

    resources: dict = {
        "training images": (
            "train-images-idx3-ubyte.gz",
            "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        ),
        "training labels": (
            "train-labels-idx1-ubyte.gz",
            "d53e105ee54ea40749a09fcbcd1e9432",
        ),
        "test images": (
            "t10k-images-idx3-ubyte.gz",
            "9fb629c4189551a2d022fa330f9573f3",
        ),
        "test labels": (
            "t10k-labels-idx1-ubyte.gz",
            "ec29112dd5afa0611ce80d1b7f02629c",
        ),
    }

    def __call__(self, *args, **kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]:
        return mnist(*args, **kwargs)


class FashionMNIST(MNIST):
    """Fashion MNIST dataset.

    References:
        [1] https://github.com/zalandoresearch/fashion-mnist
    """

    classes: tuple = (
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    )  # 10

    mirrors: list = [
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
    ]

    resources: dict = {
        "training images": (
            "train-images-idx3-ubyte.gz",
            "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
        ),
        "training labels": (
            "train-labels-idx1-ubyte.gz",
            "25c81989df183df01b3e8a0aad5dffbe",
        ),
        "test images": (
            "t10k-images-idx3-ubyte.gz",
            "bef4ecab320f06d8554ea6380940ec79",
        ),
        "test labels": (
            "t10k-labels-idx1-ubyte.gz",
            "bb300cfdad3c16e7a12a480ee83cd310",
        ),
    }

    def __call__(self, *args, **kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]:
        return fashion_mnist(*args, **kwargs)


class KuzushijiMNIST(MNIST):
    """Kuzushiji-MNIST dataset.

    References:
        [1] https://github.com/rois-codh/kmnist
    """

    classes = (
        "o",
        "ki",
        "su",
        "tsu",
        "na",
        "ha",
        "ma",
        "ya",
        "re",
        "wo",
    )  # 10

    mirrors = ["http://codh.rois.ac.jp/kmnist/dataset/kmnist/"]

    resources = {
        "training images": (
            "train-images-idx3-ubyte.gz",
            "bdb82020997e1d708af4cf47b453dcf7",
        ),
        "training labels": (
            "train-labels-idx1-ubyte.gz",
            "e144d726b3acfaa3e44228e80efcd344",
        ),
        "test images": (
            "t10k-images-idx3-ubyte.gz",
            "5c965bf0a639b31b8f53240b1b52f4d7",
        ),
        "test labels": (
            "t10k-labels-idx1-ubyte.gz",
            "7320c461ea6c1c855c0b718fb2a4b134",
        ),
    }

    def __call__(self, *args, **kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]:
        return kuzushiji_mnist(*args, **kwargs)


class CIFAR10(Dataset[Tuple[numpy.ndarray, numpy.ndarray]]):
    """CIFAR10 dataset.

    CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

    References:
        [1] https://www.cs.toronto.edu/~kriz/cifar.html
    """

    size: int = 6000 * 10

    dimensions: int = 32 * 32 * 3

    labeled: bool = True

    classes: tuple = (
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )  # 10

    mirrors: list = [
        "https://www.cs.toronto.edu/~kriz/",
    ]

    resources: tuple = (
        "cifar-10-python.tar.gz",
        "c58f30108f718f92721af3b95e74349a",
    )

    def __call__(self, *args, **kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]:
        return cifar10(*args, **kwargs)


class CIFAR100(CIFAR10):
    """CIFAR100 dataset.

    CIFAR100 dataset has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class. The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).

    References:
        [1] https://www.cs.toronto.edu/~kriz/cifar.html
    """

    size: int = 600 * 100

    dimensions: int = 32 * 32 * 3

    classes: tuple = ()  # 100

    mirrors: list = [
        "https://www.cs.toronto.edu/~kriz/",
    ]

    resources: tuple = (
        "cifar-100-python.tar.gz",
        "eb9058c3a382ffc7106e4002c42a8d85",
    )

    def __call__(self, *args, **kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]:
        return cifar100(*args, **kwargs)


class COIL20(Dataset[Tuple[numpy.ndarray, numpy.ndarray]]):
    """COIL-20 dataset.

    References:
        [1] https://cave.cs.columbia.edu/repository/COIL-20
    """

    size: int = 72 * 20

    dimensions: int = 128 * 128

    labeled: bool = True

    classes: tuple = ()  # 20

    url: str = "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip"

    def __call__(self, *args, **kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]:
        return cifar100(*args, **kwargs)


class COIL100(COIL20):
    """COIL-100 dataset.

    References:
        [1] https://cave.cs.columbia.edu/repository/COIL-100
    """

    size: int = 72 * 100

    dimensions: int = 128 * 128 * 3

    labeled: bool = True

    classes: tuple = ()  # 100

    url: str = "http://cave.cs.columbia.edu/old/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip"


class Tess(Dataset[Tuple[numpy.ndarray, numpy.ndarray]]):
    """Tess dataset from FR-FCM-ZZ36.

    References:
        [1] http://flowrepository.org/id/FR-FCM-ZZ36
    """

    size: int = 1000000

    dimensions: int = 14

    labeled: bool = False

    def __call__(self, *args, **kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]:
        return tess(*args, **kwargs)


class Anna(Tess):
    """Anna dataset from FR-FCM-ZZSC.

    References:
        [1] http://flowrepository.org/id/FR-FCM-ZZSC
    """

    size: int = 3176162

    dimensions: int = 16

    labeled: bool = False

    def __call__(self, *args, **kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]:
        return anna(*args, **kwargs)


class Levine13dim(Tess):
    """Levine_13dim dataset from FR-FCM-ZZPH.

    Bone marrow cells from healthy donor (human organism).

    References:
        [1] http://flowrepository.org/id/FR-FCM-ZZPH
    """

    size: int = 167044

    dimensions: int = 13

    labeled: bool = True

    labels: int = 81747  # (49%)

    individuals: int = 1

    classes: tuple = (
        "ungated",
        "CD11b-_Monocyte_cells",
        "CD11bhi_Monocyte_cells",
        "CD11bmid_Monocyte_cells",
        "CMP_cells",
        "Erythroblast_cells",
        "GMP_cells",
        "HSC_cells",
        "Immature_B_cells",
        "Mature_CD38lo_B_cells",
        "Mature_CD38mid_B_cells",
        "Mature_CD4+_T_cells",
        "Mature_CD8+_T_cells",
        "Megakaryocyte_cells",
        "MEP_cells",
        "MPP_cells",
        "Myelocyte_cells",
        "Naive_CD4+_T_cells",
        "Naive_CD8+_T_cells",
        "NK_cells",
        "Plasma_cell_cells",
        "Plasmacytoid_DC_cells",
        "Platelet_cells",
        "Pre-B_I_cells",
        "Pre-B_II_cells",
    )  # 24

    def __call__(self, *args, **kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]:
        return levine13dim(*args, **kwargs)


class Levine32dim(Levine13dim):
    """Levine_32dim dataset from FR-FCM-ZZPH.

    Bone marrow cells from healthy donors (human organism).

    References:
        [1] http://flowrepository.org/id/FR-FCM-ZZPH
    """

    size: int = 265627

    dimensions: int = 32

    labeled: bool = True

    labels: int = 104184  # (39%)

    individuals: int = 2

    classes: tuple = (
        "ungated",
        "Basophils",
        "CD16-_NK_cells",
        "CD16+_NK_cells",
        "CD34+CD38+CD123-_HSPCs",
        "CD34+CD38+CD123+_HSPCs",
        "CD34+CD38lo_HSCs",
        "CD4_T_cells",
        "CD8_T_cells",
        "Mature_B_cells",
        "Monocytes",
        "pDCs",
        "Plasma_B_cells",
        "Pre_B_cells",
        "Pro_B_cells",
    )  # 14

    def __call__(self, *args, **kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]:
        return levine32dim(*args, **kwargs)


class Samusik01(Levine13dim):
    """Samusik_01 dataset from FR-FCM-ZZPH.

    Replicate bone marrow samples from C57BL/6J mice (individual 01 only, mouse organism).

    References:
        [1] http://flowrepository.org/id/FR-FCM-ZZPH
    """

    size: int = 86864

    dimensions: int = 39

    labeled: bool = True

    labels: int = 53173  # (61%)

    individuals: int = 1

    classes: tuple = (
        "ungated",
        "B-cell Frac A-C (pro-B cells)",
        "Basophils",
        "CD4 T cells",
        "CD8 T cells",
        "Classical Monocytes",
        "CLP",
        "CMP",
        "Eosinophils",
        "gd T cells",
        "GMP",
        "HSC",
        "IgD- IgMpos B cells",
        "IgDpos IgMpos B cells",
        "IgM- IgD- B-cells",
        "Intermediate Monocytes",
        "Macrophages",
        "mDCs",
        "MEP",
        "MPP",
        "NK cells",
        "NKT cells",
        "Non-Classical Monocytes",
        "pDCs",
        "Plasma Cells",
    )  # 24

    def __call__(self, *args, **kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]:
        return samusik01(*args, **kwargs)


class Samusik(Levine13dim):
    """Samusik_all dataset from FR-FCM-ZZPH.

    Replicate bone marrow samples from C57BL/6J mice (all, mouse organism).

    References:
        [1] http://flowrepository.org/id/FR-FCM-ZZPH
    """

    size: int = 841644

    dimensions: int = 39

    labeled: bool = True

    labels: int = 514386  # (61%)

    individuals: int = 10

    classes: tuple = (
        "ungated",
        "B-cell Frac A-C (pro-B cells)",
        "Basophils",
        "CD4 T cells",
        "CD8 T cells",
        "Classical Monocytes",
        "CLP",
        "CMP",
        "Eosinophils",
        "gd T cells",
        "GMP",
        "HSC",
        "IgD- IgMpos B cells",
        "IgDpos IgMpos B cells",
        "IgM- IgD- B-cells",
        "Intermediate Monocytes",
        "Macrophages",
        "mDCs",
        "MEP",
        "MPP",
        "NK cells",
        "NKT cells",
        "Non-Classical Monocytes",
        "pDCs",
        "Plasma Cells",
    )  # 24

    def __call__(self, *args, **kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]:
        return samusik(*args, **kwargs)


class Nilsson(Levine13dim):
    """Nilsson_rare dataset from FR-FCM-ZZPH.

    Bone marrow cells from healthy donor (human organism).

    References:
        [1] http://flowrepository.org/id/FR-FCM-ZZPH
    """

    size: int = 44140

    dimensions: int = 13

    labeled: bool = True

    labels: int = 358  # (0.8%)

    individuals: int = 1

    classes: tuple = (
        "ungated",
        "hematopoietic stem cells",
    )  # 1

    def __call__(self, *args, **kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]:
        return nilsson(*args, **kwargs)


class Mosmann(Levine13dim):
    """Mosmann_rare dataset from FR-FCM-ZZPH.

    Peripheral blood cells from healthy donor, stimulated with influenza antigens (human organism).

    References:
        [1] http://flowrepository.org/id/FR-FCM-ZZPH
    """

    size: int = 396460

    dimensions: int = 14

    labeled: bool = True

    labels: int = 109  # (0.03 %)

    individuals: int = 1

    classes: tuple = (
        "ungated",
        "activated memory CD4 T cells",
    )  # 1

    def __call__(self, *args, **kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]:
        return mosmann(*args, **kwargs)
