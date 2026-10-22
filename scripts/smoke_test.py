"""Run a mapper-only smoke test without data, training, or detector weights."""

from __future__ import annotations

import argparse

import torch

from wsm_hsi import NNPMapper, WSMMapper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, choices=(16, 96, 112), required=True)
    args = parser.parse_args()

    x = torch.rand(2, args.channels, 32, 32)
    nnp = NNPMapper(args.channels)
    wsm = WSMMapper(args.channels)
    nnp_output = nnp(x)
    wsm_output, auxiliary = wsm(x)

    assert nnp_output.shape == (2, 3, 32, 32)
    assert wsm_output.shape == (2, 3, 32, 32)
    assert torch.isfinite(wsm_output).all()
    assert all(torch.isfinite(value).all() for value in auxiliary.values())
    print(f"PASS channels={args.channels} output_shape={tuple(wsm_output.shape)}")


if __name__ == "__main__":
    main()
