# Cross-Family Latency Profiling of Object Detectors on an NVIDIA Tesla T4

Operator-level latency profiling and FP16 / TensorRT optimisation of **24 object detectors**
across 5 architecture families, benchmarked under a single protocol on one NVIDIA Tesla T4.

📄 **[Read the paper](./Results_Paper.pdf)** · Télécom Paris, M1 DSAI, May 2026

---

## The question

The standard way to compare detectors is mAP for accuracy and FLOPs or FPS for speed. Neither
tells you *why* a model runs at the speed it does. FLOPs ignore memory bandwidth, the per-kernel
launch overhead that accumulates at batch 1, and data-dependent operations like NMS whose cost
changes with the input.

So instead of producing another ranking, we broke each model's latency down into its component
operators — convolution, normalisation, activation, attention, post-processing — and asked which
part is actually responsible for the cost, and why.

## Protocol

One 640×640 COCO *val2017* image at a time, FP32 eager mode, **batch size 1** — the hardest case,
since there is no batching to hide launch cost. 50 warm-up iterations discarded, then 1,000 timed
forward passes bracketed by `torch.cuda.Event` around `torch.cuda.synchronize()`. Median latency
reported. Accuracy is COCO mAP<sup>50:95</sup> via `pycocotools` over the full validation set.
Seed fixed at 42. Profiling with `torch.profiler` and NVIDIA Nsight Systems.

## What we found

**Convolution dominates everywhere** — 56% to 90% of backbone time — and runs in FP32 with the
T4's Tensor Cores completely idle. Mixed precision is therefore the single largest lever.

**Normalisation is the clearest free win.** The DETR family loses 28–32% of its backbone to a
`FrozenBatchNorm2d` that `fuse_conv_bn_eval` does not recognise, so it runs as separate
element-wise kernels. Folded by hand, it costs nothing.

**One missing kernel can cost an order of magnitude.** Deformable DETR is the slowest model in the
study at 263 ms, not because of its architecture but because the deformable-attention operator has
no compiled CUDA kernel in this build and falls back to a Python loop. An operator, not a parameter
count, sets latency.

## Results

Three mechanisms produce the gains, in order of payoff:

| Mechanism | What it does | Best measured |
|---|---|---|
| **Precision** (FP16 autocast) | Moves convolution and GEMM onto the Tensor Cores | 6.62× on GEMM (RT-DETR R101) |
| **Fusion** (Conv-BN-activation) | Deletes bandwidth-bound kernels as convolution epilogues | 4.88× on SiLU (YOLOv7-X) |
| **Engine capture** (TensorRT) | Collapses hundreds of per-image launches into one | 11.5× fewer launches (RT-DETR R101) |

End-to-end: **1.16× to 2.34×** from FP16 autocast alone across all families, **5.06×** and **5.57×**
on RT-DETR R50/R101 under TensorRT FP16 — with **mAP unchanged** throughout.

The floor is equally consistent. Selection, NMS, RoIAlign and the deformable-attention fallback are
data-dependent or un-compiled: no precision or fusion lever reaches them. That ranking is then read
forwards, as a design brief for a detector assembled from operations that accelerate well.

## My contribution

Single team report aggregating six individual profiling and optimisation studies.

I was responsible for the **YOLOX / DAMO-YOLO / YOLOv7 family** — six models, from FP32 baseline
profiling through to the TensorRT engine:

| Model | FP32 | Optimised | Speed-up |
|---|---|---|---|
| YOLOX-m | 21.8 ms | 18.8 ms | 1.16× |
| YOLOX-l | 41.8 ms | 22.0 ms | 1.90× |
| YOLOX-x | 75.6 ms | 33.4 ms | 2.26× |
| YOLOv7 | 33.9 ms | 17.9 ms | 1.90× |
| YOLOv7-X | 60.7 ms | 26.6 ms | 2.29× |
| DAMO-YOLO-M | 25.3 ms | 15.0 ms | 1.69× |

At the operator level under TensorRT FP16: **5.00×** on total convolution time (YOLOv7) and
**4.88×** on fused SiLU (YOLOv7-X) — the strongest convolution acceleration in the study.

Two negative results worth as much as the positive ones. `torch.compile` could not trace YOLOv7's
flat `nn.Sequential` (no module hierarchy to fuse across), and CUDA-graph capture broke on
YOLOX-m's dynamic control flow. Both failures are reported and explained rather than omitted,
and they are why the family's per-submodule gains come from AMP and TensorRT instead.

## Team

Two-stage R-CNN (Charles Kayssieh) · DETR family and shared tooling (Antoine Besson) ·
RT-DETR family (Vansh) · **YOLOX / DAMO / YOLOv7 (Jesus Sebastian Lamas)** ·
YOLOv5-11s (Yassine) · RetinaNet, FCOS, EfficientDet (Pacome)

Télécom Paris, Institut Polytechnique de Paris — M1 DSAI, May 2026
