# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260405_224845/code_0.py
# fused_model.py
# -------------------------------------------------------------
# Optimised ModelNew – linear layer fused with row‑wise sum.
# -------------------------------------------------------------
#
# The original implementation performed:
#   y = x @ Wᵀ + b                # dense GEMM + bias
#   out[i] = Σ_k y[i, k]          # row‑wise reduction
#
# Because the reduction is a sum over the output dimension, the whole
# computation can be expressed analytically as:
#   out_i = x_i · (W.sum(dim=0)) + b.sum()
#
# This implementation therefore:
#   * Pre‑computes the per‑output sums of the weight matrix and bias.
#   * Uses a single matrix‑vector product (torch.matmul) per forward.
#   * Keeps the public API identical – returns a tensor of shape (B, 1).
#
# The code is completely self‑contained in a single Python file,
# requires no custom CUDA kernels and runs entirely on the GPU via
# cuBLAS (torch.matmul).  The weight/bias sums are stored as buffers
# so that they move with the module and stay on the correct device.
#
# -------------------------------------------------------------

import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """
    Linear layer followed by a row‑wise sum.
    The two ops are fused into a single dot‑product with a pre‑computed
    weight vector `w_sum = W.sum(dim=0)` and a scalar bias `b_sum = b.sum()`.
    The forward pass therefore performs:
        out = x @ w_sum + b_sum
    and returns shape (B, 1) to preserve the original API.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

        # ------------------------------------------------------------------
        # Compute (and store) the fused weight and bias sums.
        # They are registered as buffers so they travel with the module.
        # ------------------------------------------------------------------
        self.register_buffer(
            "w_sum",                      # shape: (in_features,)
            self.linear.weight.sum(dim=0)  # sum over output dimension
        )
        self.register_buffer(
            "b_sum",                      # shape: (1,)
            self.linear.bias.sum().unsqueeze(0)
        )

        # ------------------------------------------------------------------
        # Flag that tells us when the cached sums need recomputation.
        # We use a hook on the weight and bias that flips the flag whenever
        # a gradient update modifies them.
        # ------------------------------------------------------------------
        self._sums_stale = False
        self.linear.weight.register_hook(self._mark_stale)
        if self.linear.bias is not None:
            self.linear.bias.register_hook(self._mark_stale)

    # ------------------------------------------------------------------
    # Hook called during the backward pass; marks cached sums as stale.
    # ------------------------------------------------------------------
    def _mark_stale(self, grad):
        self._sums_stale = True
        return grad

    # ------------------------------------------------------------------
    # Update cached sums if they have become stale.
    # This is a cheap O(in_features + out_features) reduction,
    # negligible compared with the original O(B·in_features·out_features) work.
    # ------------------------------------------------------------------
    def _update_sums_if_needed(self):
        if self._sums_stale:
            # .detach() because we do not want these reductions to be tracked.
            self.w_sum.copy_(self.linear.weight.sum(dim=0).detach())
            self.b_sum.copy_(self.linear.bias.sum().detach().unsqueeze(0))
            self._sums_stale = False

    # ------------------------------------------------------------------
    # Forward – fused operation.
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, in_features) on CUDA.

        Returns:
            Tensor of shape (B, 1) on CUDA.
        """
        # Ensure the cached sums reflect the latest parameters.
        self._update_sums_if_needed()

        # (B, in_features) @ (in_features,) -> (B,)
        out = torch.matmul(x, self.w_sum) + self.b_sum  # scalar broadcasted

        # Preserve original API shape (B, 1)
        return out.unsqueeze(-1)

batch_size = 1024
in_features = 8192
out_features = 8192


# ------------------------------------------------------------------
# Helper functions required by the test harness.
# The harness injects `batch_size`, `in_features` and `out_features`
# into the global namespace before calling them.
# ------------------------------------------------------------------
def get_inputs():
    """Return a list containing a single input tensor on CUDA."""
    return [torch.rand(batch_size, in_features, device="cuda")]


def get_init_inputs():
    """Return the arguments required to construct ModelNew."""
    return [in_features, out_features]


# ------------------------------------------------------------------
# Simple sanity‑check (runs when the file is executed directly).
# ------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # Example configuration (the harness will overwrite these globals).
    model = ModelNew(in_features, out_features).cuda()
    model.eval()
    with torch.no_grad():
        inp = torch.rand(batch_size, in_features, device="cuda")
        out = model(inp)
        print(f"output shape: {out.shape}, mean value: {out.mean().item():.6f}")

