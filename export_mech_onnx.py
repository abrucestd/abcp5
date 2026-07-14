#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import torch

from train3_mech import IN_DIM, Regressor43D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="reg43d_mech_v0.4.pt")
    parser.add_argument("--onnx", default="model4.onnx")
    parser.add_argument("--scale", default="model4_scale.txt")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = Regressor43D()
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.zeros(1, IN_DIM, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        args.onnx,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )

    x_mean = ckpt["x_mean"].detach().cpu().float().view(-1)
    x_std = ckpt["x_std"].detach().cpu().float().view(-1)
    if x_mean.numel() != IN_DIM or x_std.numel() != IN_DIM:
        raise ValueError(f"bad stats shape: x_mean={x_mean.numel()}, x_std={x_std.numel()}, IN_DIM={IN_DIM}")

    vals = [float(ckpt["y_scale"]), float(ckpt.get("y_bias", 0.0))]
    vals += [float(v) for v in x_mean]
    vals += [float(v if abs(float(v)) > 1e-12 else 1.0) for v in x_std]
    Path(args.scale).write_text(" ".join(f"{v:.9g}" for v in vals) + "\n", encoding="utf-8")

    print(f"[INFO] exported {args.onnx}")
    print(f"[INFO] exported {args.scale}, values={len(vals)}")


if __name__ == "__main__":
    main()
