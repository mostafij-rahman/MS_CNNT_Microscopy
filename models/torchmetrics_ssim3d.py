import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce
from typing import Any, List, Optional, Sequence, Tuple, Union, Literal


def _gaussian_kernel_3d(channel: int, kernel_size: Sequence[int], sigma: Sequence[float], dtype: torch.dtype, device: torch.device) -> Tensor:
    """Creates a 3D Gaussian kernel."""
    coords = [torch.arange(size, dtype=dtype, device=device) for size in kernel_size]
    coords = [coord - (size - 1) / 2 for coord, size in zip(coords, kernel_size)]
    kernel = torch.exp(-0.5 * sum((coord ** 2 / s ** 2 for coord, s in zip(coords, sigma))))
    kernel = kernel / kernel.sum()
    kernel = kernel.expand(channel, 1, *kernel_size)
    return kernel


def _reflection_pad_3d(x: Tensor, pad_d: int, pad_w: int, pad_h: int) -> Tensor:
    """Pads a 5D tensor with reflection padding."""
    return F.pad(x, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d), mode='reflect')


def _ssim_check_inputs(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Update and returns variables required to compute Structural Similarity Index Measure."""
    if preds.dtype != target.dtype:
        target = target.to(preds.dtype)
    _check_same_shape(preds, target)
    if len(preds.shape) not in (4, 5):
        raise ValueError(
            "Expected `preds` and `target` to have BxCxHxW or BxCxDxHxW shape."
            f" Got preds: {preds.shape} and target: {target.shape}."
        )
    return preds, target


def _ssim_update(
    preds: Tensor,
    target: Tensor,
    gaussian_kernel: bool = True,
    sigma: Union[float, Sequence[float]] = 1.5,
    kernel_size: Union[int, Sequence[int]] = 11,
    data_range: Optional[Union[float, Tuple[float, float]]] = None,
    k1: float = 0.01,
    k2: float = 0.03,
    return_full_image: bool = False,
    return_contrast_sensitivity: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Compute Structural Similarity Index Measure."""
    is_3d = preds.ndim == 5

    if not isinstance(kernel_size, Sequence):
        kernel_size = 3 * [kernel_size] if is_3d else 2 * [kernel_size]
    if not isinstance(sigma, Sequence):
        sigma = 3 * [sigma] if is_3d else 2 * [sigma]

    if len(kernel_size) != len(target.shape) - 2:
        raise ValueError(
            f"`kernel_size` has dimension {len(kernel_size)}, but expected to be two less that target dimensionality,"
            f" which is: {len(target.shape)}"
        )
    if len(kernel_size) not in (2, 3):
        raise ValueError(
            f"Expected `kernel_size` dimension to be 2 or 3. `kernel_size` dimensionality: {len(kernel_size)}"
        )
    if len(sigma) != len(target.shape) - 2:
        raise ValueError(
            f"`kernel_size` has dimension {len(kernel_size)}, but expected to be two less that target dimensionality,"
            f" which is: {len(target.shape)}"
        )
    if len(sigma) not in (2, 3):
        raise ValueError(
            f"Expected `kernel_size` dimension to be 2 or 3. `kernel_size` dimensionality: {len(kernel_size)}"
        )

    if return_full_image and return_contrast_sensitivity:
        raise ValueError("Arguments `return_full_image` and `return_contrast_sensitivity` are mutually exclusive.")

    if any(x % 2 == 0 or x <= 0 for x in kernel_size):
        raise ValueError(f"Expected `kernel_size` to have odd positive number. Got {kernel_size}.")

    if any(y <= 0 for y in sigma):
        raise ValueError(f"Expected `sigma` to have positive number. Got {sigma}.")

    if data_range is None:
        data_range = max(preds.max() - preds.min(), target.max() - target.min())  # type: ignore[call-overload]
    elif isinstance(data_range, tuple):
        preds = torch.clamp(preds, min=data_range[0], max=data_range[1])
        target = torch.clamp(target, min=data_range[0], max=data_range[1])
        data_range = data_range[1] - data_range[0]

    c1 = pow(k1 * data_range, 2)  # type: ignore[operator]
    c2 = pow(k2 * data_range, 2)  # type: ignore[operator]
    device = preds.device

    channel = preds.size(1)
    dtype = preds.dtype
    gauss_kernel_size = [int(3.5 * s + 0.5) * 2 + 1 for s in sigma]

    pad_h = (gauss_kernel_size[0] - 1) // 2
    pad_w = (gauss_kernel_size[1] - 1) // 2

    if is_3d:
        pad_d = (gauss_kernel_size[2] - 1) // 2
        preds = _reflection_pad_3d(preds, pad_d, pad_w, pad_h)
        target = _reflection_pad_3d(target, pad_d, pad_w, pad_h)
        if gaussian_kernel:
            kernel = _gaussian_kernel_3d(channel, gauss_kernel_size, sigma, dtype, device)
    else:
        preds = F.pad(preds, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
        target = F.pad(target, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
        if gaussian_kernel:
            kernel = _gaussian_kernel_2d(channel, gauss_kernel_size, sigma, dtype, device)

    if not gaussian_kernel:
        kernel = torch.ones((channel, 1, *kernel_size), dtype=dtype, device=device) / torch.prod(
            torch.tensor(kernel_size, dtype=dtype, device=device)
        )

    input_list = torch.cat((preds, target, preds * preds, target * target, preds * target))  # (5 * B, C, H, W)

    outputs = F.conv3d(input_list, kernel, groups=channel) if is_3d else F.conv2d(input_list, kernel, groups=channel)

    output_list = outputs.split(preds.shape[0])

    mu_pred_sq = output_list[0].pow(2)
    mu_target_sq = output_list[1].pow(2)
    mu_pred_target = output_list[0] * output_list[1]

    # Calculate the variance of the predicted and target images, should be non-negative
    sigma_pred_sq = torch.clamp(output_list[2] - mu_pred_sq, min=0.0)
    sigma_target_sq = torch.clamp(output_list[3] - mu_target_sq, min=0.0)
    sigma_pred_target = output_list[4] - mu_pred_target

    upper = 2 * sigma_pred_target.to(dtype) + c2
    lower = (sigma_pred_sq + sigma_target_sq).to(dtype) + c2

    ssim_idx_full_image = ((2 * mu_pred_target + c1) * upper) / ((mu_pred_sq + mu_target_sq + c1) * lower)

    if is_3d:
        ssim_idx = ssim_idx_full_image[..., pad_h:-pad_h, pad_w:-pad_w, pad_d:-pad_d]
    else:
        ssim_idx = ssim_idx_full_image[..., pad_h:-pad_h, pad_w:-pad_w]

    if return_contrast_sensitivity:
        contrast_sensitivity = upper / lower
        if is_3d:
            contrast_sensitivity = contrast_sensitivity[..., pad_h:-pad_h, pad_w:-pad_w, pad_d:-pad_d]
        else:
            contrast_sensitivity = contrast_sensitivity[..., pad_h:-pad_h, pad_w:-pad_w]
        return ssim_idx.reshape(ssim_idx.shape[0], -1).mean(-1), contrast_sensitivity.reshape(
            contrast_sensitivity.shape[0], -1
        ).mean(-1)

    if return_full_image:
        return ssim_idx.reshape(ssim_idx.shape[0], -1).mean(-1), ssim_idx_full_image

    return ssim_idx.reshape(ssim_idx.shape[0], -1).mean(-1)


def _ssim_compute(
    similarities: Tensor,
    reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
) -> Tensor:
    """Apply the specified reduction to pre-computed structural similarity."""
    return reduce(similarities, reduction)

class StructuralSimilarityIndexMeasure3D(Metric):
    """Compute Structural Similarity Index Measure (SSIM) for 3D images.

    Args:
        gaussian_kernel: If ``True`` (default), a gaussian kernel is used, if ``False`` a uniform kernel is used
        sigma: Standard deviation of the gaussian kernel, anisotropic kernels are possible.
            Ignored if a uniform kernel is used
        kernel_size: the size of the uniform kernel, anisotropic kernels are possible.
            Ignored if a Gaussian kernel is used
        reduction: a method to reduce metric score over individual batch scores

            - ``'elementwise_mean'``: takes the mean
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

        data_range:
            the range of the data. If None, it is determined from the data (max - min). If a tuple is provided then
            the range is calculated as the difference and input is clamped between the values.
        k1: Parameter of SSIM.
        k2: Parameter of SSIM.
        return_full_image: If true, the full ``ssim`` image is returned as a second argument.
            Mutually exclusive with ``return_contrast_sensitivity``
        return_contrast_sensitivity: If true, the constant term is returned as a second argument.
            The luminance term can be obtained with luminance=ssim/contrast
            Mutually exclusive with ``return_full_image``
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> import torch
        >>> from torchmetrics.image import StructuralSimilarityIndexMeasure3D
        >>> preds = torch.rand([3, 1, 32, 256, 256])
        >>> target = preds * 0.75
        >>> ssim = StructuralSimilarityIndexMeasure3D(data_range=1.0)
        >>> ssim(preds, target)
        tensor(0.9219)

    """

    higher_is_better: bool = True
    is_differentiable: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        gaussian_kernel: bool = True,
        sigma: Union[float, Sequence[float]] = 1.5,
        kernel_size: Union[int, Sequence[int]] = 11,
        reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
        data_range: Optional[Union[float, Tuple[float, float]]] = None,
        k1: float = 0.01,
        k2: float = 0.03,
        return_full_image: bool = False,
        return_contrast_sensitivity: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        valid_reduction = ("elementwise_mean", "sum", "none", None)
        if reduction not in valid_reduction:
            raise ValueError(f"Argument `reduction` must be one of {valid_reduction}, but got {reduction}")

        if reduction in ("elementwise_mean", "sum"):
            self.add_state("similarity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        else:
            self.add_state("similarity", default=[], dist_reduce_fx="cat")

        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        if return_contrast_sensitivity or return_full_image:
            self.add_state("image_return", default=[], dist_reduce_fx="cat")

        self.gaussian_kernel = gaussian_kernel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2
        self.return_full_image = return_full_image
        self.return_contrast_sensitivity = return_contrast_sensitivity

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        preds, target = _ssim_check_inputs(preds, target)
        similarity_pack = _ssim_update(
            preds,
            target,
            self.gaussian_kernel,
            self.sigma,
            self.kernel_size,
            self.data_range,
            self.k1,
            self.k2,
            self.return_full_image,
            self.return_contrast_sensitivity,
        )

        if isinstance(similarity_pack, tuple):
            similarity, image = similarity_pack
        else:
            similarity = similarity_pack

        if self.return_contrast_sensitivity or self.return_full_image:
            self.image_return.append(image)

        if self.reduction in ("elementwise_mean", "sum"):
            self.similarity += similarity.sum()
            self.total += preds.shape[0]
        else:
            self.similarity.append(similarity)

    def compute(self) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Compute SSIM over state."""
        if self.reduction == "elementwise_mean":
            similarity = self.similarity / self.total
        elif self.reduction == "sum":
            similarity = self.similarity
        else:
            similarity = torch.cat(self.similarity, dim=0)

        if self.return_contrast_sensitivity or self.return_full_image:
            image_return = torch.cat(self.image_return, dim=0)
            return similarity, image_return

        return similarity

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[Any] = None
    ) -> Any:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed
        """
        return self._plot(val, ax)
