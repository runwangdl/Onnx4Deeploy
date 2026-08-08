# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Convolutional LoRA with *loralib* semantics.

This reproduces the ``loralib.ConvLoRA`` formulation (microsoft/LoRA), i.e. the
delta weight is **materialised first and the convolution is run once**::

    y = conv(x, W + (B @ A).view(W.shape) * scaling)

rather than the "two convolutions" formulation used by PEFT
(``y = conv(x, W) + conv_B(conv_A(x))``).

Matricisation (this is the part that differs from every other LoRA impl)
-----------------------------------------------------------------------
``loralib`` folds *one* kernel dimension into the input side and *one* into the
output side of the factorisation::

    A in R^{(r * k) x (C_in * k)}
    B in R^{(C_out / groups * k) x (r * k)}

so the low-rank decomposition is taken over the ``(C_out*k) x (C_in*k)``
matricisation of the kernel and its rank bound is ``r * k`` -- **not** ``r``.

Concretely, ``Conv2d(1, 16, kernel_size=3, r=2)`` gives ``lora_A: [6, 3]`` and
``lora_B: [48, 6]``.

Consequence for ablations: with ``r=8, k=3`` the *effective rank* is ``24``.
A PEFT ``r=8`` conv adapter has effective rank ``8``.  The two are **not**
comparable at equal ``r`` -- to compare like for like, either match ``r*k``
against PEFT's ``r``, or match parameter counts explicitly.

Note that ``scaling = lora_alpha / r`` (divided by ``r``, *not* by ``r * k``);
this too is kept bit-for-bit identical to ``loralib``.

Limitation: ``groups > 1``
--------------------------
The upstream ``loralib`` implementation is **silently incorrect** for grouped /
depthwise convolutions, and this module therefore rejects them by default.

It is worth being precise about the failure mode, because it is *not* a crash.
The element counts happen to agree for every ``groups``::

    numel(B @ A) = (C_out/g * k) * (C_in * k) = C_out * C_in * k^2 / g
    numel(W)     =  C_out * (C_in/g) * k^2    = C_out * C_in * k^2 / g

so ``.view(W.shape)`` **succeeds** and returns a tensor of the right shape.  But
``A`` spans the *full* ``C_in`` while ``W`` only holds ``C_in/g`` input channels
per group, so the reshape scatters delta entries across group boundaries: the
adapter for group ``i`` bleeds into group ``j``.  The result is a wrong-but-
plausible model that trains and exports without ever raising.

Because it fails quietly rather than loudly, this module asserts instead.  Pass
``allow_grouped=True`` to opt into the upstream behaviour knowingly.
"""

import math
from typing import Type

import torch
import torch.nn as nn

__all__ = ["ConvLoRA", "Conv1d", "Conv2d", "lora_parameter_names", "mark_only_lora_as_trainable"]


class ConvLoRA(nn.Module):
    """Base class wrapping a conv module with a loralib-style low-rank adapter.

    Args:
        conv_module: The ``nn.Module`` *class* to instantiate (``nn.Conv1d`` /
            ``nn.Conv2d``).  Subclasses bind this.
        in_channels: Conv input channels.
        out_channels: Conv output channels.
        kernel_size: Kernel size.  Must be an ``int`` (loralib restriction --
            the matricisation is only defined for square/symmetric kernels).
        r: LoRA rank *parameter*.  The realised rank bound is ``r * kernel_size``.
        lora_alpha: LoRA scaling numerator; ``scaling = lora_alpha / r``.
        lora_dropout: Dropout applied to the input of the LoRA path.  Note that
            loralib applies dropout to ``x``, but since this formulation folds
            the adapter into the *weight*, dropout cannot be applied to ``x``
            for the LoRA branch alone; it is applied to the whole input, exactly
            as upstream does.  Defaults to 0 (recommended for ONNX export).
        merge_weights: If True, ``eval()`` folds the delta into ``conv.weight``
            and ``train()`` unfolds it.  Defaults to **False** here (upstream
            defaults to True) because merging during an ``eval()`` call would
            erase the LoRA branch from a traced ONNX graph.
        allow_grouped: Opt in to the (broken) ``groups > 1`` behaviour.
        b_init: ``"zeros"`` reproduces loralib exactly.  ``"small_normal"``
            seeds ``lora_B`` with a small normal instead -- useful when a
            downstream numerical test needs a non-degenerate gradient for
            ``lora_A`` on the very first step (with ``B = 0`` the gradient of
            ``A`` is identically zero at step 0).
        **kwargs: Forwarded to the conv constructor (``stride``, ``padding``,
            ``bias``, ``groups``, ...).
    """

    def __init__(
        self,
        conv_module: Type[nn.Module],
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
        allow_grouped: bool = False,
        b_init: str = "zeros",
        **kwargs,
    ):
        super(ConvLoRA, self).__init__()

        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)

        if not isinstance(kernel_size, int):
            raise TypeError(
                f"ConvLoRA requires an int kernel_size (loralib matricisation is only "
                f"defined for symmetric kernels); got {kernel_size!r}."
            )

        groups = self.conv.groups
        if groups != 1 and not allow_grouped:
            raise ValueError(
                f"ConvLoRA does not support groups={groups} (got in_channels={in_channels}, "
                f"out_channels={out_channels}).\n"
                f"The loralib matricisation puts the full C_in on the input side of lora_A "
                f"while a grouped conv weight only holds C_in/groups per group. The element "
                f"counts coincidentally match, so .view() does NOT raise -- it silently "
                f"scatters delta weights across group boundaries and yields a wrong model.\n"
                f"Pass allow_grouped=True only if you knowingly want that upstream behaviour."
            )

        self.r = r
        self.lora_alpha = lora_alpha
        self.merge_weights = merge_weights
        self.merged = False
        self.kernel_size = kernel_size
        self.b_init = b_init

        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        if r > 0:
            # The matricisation folds exactly ONE kernel dimension into each side, so
            # numel(B @ A) = (C_out/g * k) * (C_in * k) = C_out * C_in * k^2 / g.
            # A conv weight has numel = C_out * C_in/g * k^spatial_dims.  These agree
            # only when spatial_dims == 2, or when k == 1.  For Conv1d with k > 1 (and
            # Conv3d with k > 1) upstream loralib raises a bare RuntimeError from
            # .view() on the first forward pass; check it up front instead.
            spatial_dims = self.conv.weight.dim() - 2
            delta_numel = (out_channels // groups * kernel_size) * (in_channels * kernel_size)
            weight_numel = self.conv.weight.numel()
            if delta_numel != weight_numel:
                raise ValueError(
                    f"{type(self).__name__}: the loralib matricisation does not close for a "
                    f"{spatial_dims}D convolution with kernel_size={kernel_size}.\n"
                    f"  numel(lora_B @ lora_A) = (C_out/g * k) * (C_in * k) = {delta_numel}\n"
                    f"  numel(conv.weight)     = {weight_numel}  (shape "
                    f"{tuple(self.conv.weight.shape)})\n"
                    f"It folds one kernel dim into each side (k^2 total), which matches a 2D "
                    f"kernel only. Conv1d/Conv3d are supported at kernel_size=1 only; upstream "
                    f"loralib fails here with a RuntimeError from .view() at forward time."
                )

            # A in R^{(r*k) x (C_in*k)},  B in R^{(C_out/groups*k) x (r*k)}
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
                self.conv.weight.new_zeros((out_channels // groups * kernel_size, r * kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freeze the pre-trained weight matrix.  NOTE: for ONNX/ORT export this
            # flag alone is NOT enough -- the export must additionally list only the
            # LoRA parameters in requires_grad, otherwise ORT still builds gradient
            # accumulators for the base weights.
            self.conv.weight.requires_grad = False

        self.reset_parameters()

    # ------------------------------------------------------------------ #

    @property
    def effective_rank(self) -> int:
        """Rank bound of the materialised delta: ``r * kernel_size`` (not ``r``)."""
        return self.r * self.kernel_size

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, "lora_A"):
            # Init A like nn.Linear's default, B at zero -> delta starts at 0.
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            if self.b_init == "zeros":
                nn.init.zeros_(self.lora_B)
            elif self.b_init == "small_normal":
                nn.init.normal_(self.lora_B, mean=0.0, std=1e-3)
            else:
                raise ValueError(f"Unknown b_init={self.b_init!r}; use 'zeros' or 'small_normal'.")

    def delta_weight(self) -> torch.Tensor:
        """The materialised ``(B @ A).view(W.shape) * scaling``."""
        return (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling

    def train(self, mode: bool = True):
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.conv.weight.data -= self.delta_weight()
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    self.conv.weight.data += self.delta_weight()
                self.merged = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                self.lora_dropout(x),
                self.conv.weight + self.delta_weight(),
                self.conv.bias,
            )
        return self.conv(x)

    def extra_repr(self) -> str:
        if self.r > 0:
            return f"r={self.r}, effective_rank={self.effective_rank}, alpha={self.lora_alpha}"
        return "r=0 (LoRA disabled)"


class Conv2d(ConvLoRA):
    """``nn.Conv2d`` with a loralib-style LoRA adapter."""

    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)


class Conv1d(ConvLoRA):
    """``nn.Conv1d`` with a loralib-style LoRA adapter.

    Only ``kernel_size=1`` is representable -- see the numel guard in
    :class:`ConvLoRA`.  The loralib matricisation assumes a 2D kernel, so for
    ``k > 1`` the factorisation has ``k`` times too many elements to reshape into
    a 1D conv weight.  Upstream this surfaces as a ``RuntimeError`` from
    ``.view()`` on the first forward; here it is raised at construction time.
    """

    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)


# ---------------------------------------------------------------------- #
# Helpers                                                                 #
# ---------------------------------------------------------------------- #

_LORA_SUFFIXES = ("lora_A", "lora_B")


def lora_parameter_names(model: nn.Module) -> list:
    """Fully-qualified names of every LoRA parameter in ``model``.

    Feed this to the exporter's ``custom_trainable_params`` -- the parameter
    names match the ONNX initializer names produced by ``torch.onnx.export``.
    """
    return [n for n, _ in model.named_parameters() if n.split(".")[-1] in _LORA_SUFFIXES]


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    """Set ``requires_grad`` only on LoRA parameters (PyTorch-side freezing)."""
    lora_names = set(lora_parameter_names(model))
    for name, param in model.named_parameters():
        param.requires_grad = name in lora_names
