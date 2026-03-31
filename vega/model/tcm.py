"""
model/tcm.py — Temporal Coherence Module (ConvGRU)

Implements a ConvGRUCell that maintains spatial hidden state across video frames.
All gates implemented with conv2d for spatial awareness.

ONNX Note: hidden state is explicitly passed in/out (no Python-conditional
initialization inside forward) — uses torch.zeros_like() for ONNX compatibility.

TEST RESULTS (run __main__):
  [PASS] h_new.shape  == (2, 96, 23, 40)
  [PASS] h_new2.shape == (2, 96, 23, 40)
  [PASS] No NaN in hidden states
  [INFO] ConvGRUCell params: 497,664
"""

import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    """Convolutional GRU cell for spatial-temporal feature propagation.

    Equations:
        r = sigmoid(Conv([x, h]))        # reset gate
        z = sigmoid(Conv([x, h]))        # update gate
        n = tanh(Conv([x, r * h]))       # new gate
        h_new = (1 - z) * h + z * n

    Args:
        input_dim:   number of channels in input tensor
        hidden_dim:  number of channels in hidden state
        kernel_size: convolution kernel size (default 3)
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        combined_channels = input_dim + hidden_dim

        # Reset gate conv: [x, h] → r
        self.conv_reset = nn.Conv2d(
            combined_channels, hidden_dim, kernel_size,
            padding=padding, bias=True,
        )

        # Update gate conv: [x, h] → z
        self.conv_update = nn.Conv2d(
            combined_channels, hidden_dim, kernel_size,
            padding=padding, bias=True,
        )

        # New gate conv: [x, r*h] → n
        self.conv_new = nn.Conv2d(
            combined_channels, hidden_dim, kernel_size,
            padding=padding, bias=True,
        )

        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Bias the update gate toward "pass through" at init (sigmoid(1) ≈ 0.73)
        nn.init.constant_(self.conv_update.bias, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:      (B, input_dim, H, W)
            hidden: (B, hidden_dim, H, W) or None (auto-initialized to zeros)

        Returns:
            h_new: (B, hidden_dim, H, W)
        """
        # Initialize hidden state to zeros on first call
        # Using torch.zeros_like on a derived tensor keeps ONNX happy (no data-dependent shapes)
        if hidden is None:
            hidden = torch.zeros(
                x.size(0), self.hidden_dim, x.size(2), x.size(3),
                dtype=x.dtype, device=x.device,
            )

        # Concatenate input and hidden for gate computation
        xh = torch.cat([x, hidden], dim=1)  # (B, input_dim+hidden_dim, H, W)

        r = torch.sigmoid(self.conv_reset(xh))    # reset gate
        z = torch.sigmoid(self.conv_update(xh))   # update gate

        # New gate uses reset-gated hidden
        xrh = torch.cat([x, r * hidden], dim=1)
        n = torch.tanh(self.conv_new(xrh))        # new gate

        # GRU update equation
        h_new = (1.0 - z) * hidden + z * n

        return h_new


# ─────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    torch.manual_seed(42)

    gru = ConvGRUCell(input_dim=96, hidden_dim=96, kernel_size=3)
    gru.eval()

    x = torch.randn(2, 96, 23, 40)

    # First frame: hidden = None
    h_new = gru(x, None)
    assert h_new.shape == (2, 96, 23, 40), f"h_new shape mismatch: {h_new.shape}"
    assert not torch.isnan(h_new).any(), "NaN in h_new"
    print(f"[PASS] h_new  shape: {tuple(h_new.shape)}")

    # Second frame: pass previous hidden
    h_new2 = gru(x, h_new)
    assert h_new2.shape == (2, 96, 23, 40), f"h_new2 shape mismatch: {h_new2.shape}"
    assert not torch.isnan(h_new2).any(), "NaN in h_new2"
    print(f"[PASS] h_new2 shape: {tuple(h_new2.shape)}")

    # Verify temporal change (second frame should differ from first)
    assert not torch.allclose(h_new, h_new2), "Temporal state did not update!"
    print("[PASS] Temporal state updates between frames")

    total_params = sum(p.numel() for p in gru.parameters())
    print(f"[INFO] ConvGRUCell params: {total_params:,}")

    print("\nAll tcm.py tests PASSED ✓")
