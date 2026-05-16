# GPU memory optimizations for UniGradICON instance optimization

Notes on the memory fixes that took `experiments/unigrad-io/create_unigrad_io_data.py`
from CUDA OOM on an 11 GiB GPU to running cleanly at full utilization, while
keeping the official UniGradICON IO protocol (Adam, `lr=2e-5`, LNCC, 50 iters).

This doc doubles as an interview-style reference for "how do you reduce GPU
memory in PyTorch training/inference without changing the algorithm?"

## TL;DR

| Trick | What it costs | What it saves | When to use |
|---|---|---|---|
| Back up `state_dict` on CPU instead of GPU | A bit of host RAM + a host-device copy on restore | ~size of parameters (~280 MB here) | You need to revert weights per-sample (per-pair IO) |
| `zero_grad(set_to_none=True)` | Re-allocation of `.grad` on each step | ~size of parameters in `.grad` buffers + less fragmentation | Always (PyTorch >= 1.7) |
| `del optimizer` + `torch.cuda.empty_cache()` after each pair | A tiny bit of compute time | Adam moments (~2x params), reduces fragmentation | Per-sample / per-batch loops where the optimizer is short-lived |
| `del loss_tuple` (or the loss variable) inside the loop | Nothing | Retained autograd graph from the last iteration | Always in custom training loops |
| Put the **final** forward and any field extraction under `torch.no_grad()` | Nothing | Avoids building a new autograd graph during post-processing | Inference paths after a training step |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Possible small allocator overhead | Fixes fragmentation when many different allocation sizes appear | Workloads with large, varied tensors (3D vision, transformers with dynamic shapes) |

Implementation lives in
[`experiments/unigrad-io/create_unigrad_io_data.py`](../experiments/unigrad-io/create_unigrad_io_data.py),
specifically `run_io_then_extract_phi_px` and the inner loop of
`run_atlas_fiver_generation`.

## Why GPU memory mattered here

The model itself (UniGradICON parameters) is small for a 3D U-Net: about
**~280 MB**. The expensive object during instance optimization is **the
autograd graph for one forward pass**:

- GradICON forwards the registration network **twice** per call to compute
  `phi_AB` and `phi_BA`.
- It then composes the two warp fields (`phi_AB(phi_BA(I))`) for the inverse
  consistency loss, which involves another 3D grid-sample.
- Inputs are pseudo-3D volumes resized to **175³**, so every saved activation
  is large (`B x C x 175 x 175 x 175 x 4 bytes`).

Together, those saved activations easily reach **~9 GiB on an 11 GiB GPU**.
That is the "the graph is 9 GB" sentence: it is **not** the model weights, it
is the per-step activation memory PyTorch's autograd has to keep alive so it
can run `.backward()`.

This is why moving the optimizer (Adam ↔ SGD) almost doesn't matter for total
memory: Adam adds ~565 MB of momentum buffers, which is only ~6% of the cost.
The dominant cost is activations, and the dominant fix is to **stop keeping
activations alive when you don't need them**.

## The fixes, with mechanism

### 1) Back up weights on **CPU**, not on the GPU

Before:

```python
state0 = copy.deepcopy(net.state_dict())
```

`copy.deepcopy` on a state dict whose tensors are on CUDA produces a copy
**still on CUDA**. For UniGradICON that pins another ~280 MB of VRAM the whole
time the optimizer is running.

After:

```python
state0_cpu = {k: v.detach().to("cpu", copy=True) for k, v in net.state_dict().items()}
...
net.load_state_dict(state0_cpu)   # PyTorch handles the device transfer on copy
```

The backup lives in host RAM. `load_state_dict` copies values back into the
existing CUDA parameters when you restore. Tradeoff: one extra
device→host→device copy per sample. With Linux pinned-memory paging this is
fast relative to the IO loop itself.

Variant: you can also stash the weights into a temporary file or a single
flat tensor; same idea.

### 2) `zero_grad(set_to_none=True)`

Before:

```python
opt.zero_grad()       # fills .grad with zeros, keeps the tensor allocated
```

After:

```python
opt.zero_grad(set_to_none=True)   # actually drops .grad
```

Two effects:

1. Lower peak memory: `.grad` is freed between steps instead of held at the
   size of the parameter. On a 280 MB model that is ~280 MB you stop paying.
2. Less fragmentation: PyTorch's caching allocator gets the block back into
   its pool and can reuse it for a different-sized allocation next iter.

Interview soundbite: *"It's the default in modern PyTorch optimizers because
it both saves memory and lets the allocator behave better, especially in
mixed-shape workloads."*

### 3) Drop the optimizer + activations as soon as you're done

In a per-pair IO loop, the optimizer only lives for the duration of one pair:

```python
opt = torch.optim.Adam(net.parameters(), lr=lr)
for _ in range(steps):
    opt.zero_grad(set_to_none=True)
    loss_tuple = net(source, target)
    loss_tuple[0].backward()
    opt.step()
    del loss_tuple              # release the autograd graph from this step
del opt                         # release Adam's 2 momentum tensors per param
torch.cuda.empty_cache()        # return freed blocks to the allocator
```

Why each line matters:

- `del loss_tuple` — `loss_tuple` holds references to GradICON's internal
  tensors (`phi_AB_vectorfield`, warped images, etc.), which the autograd
  graph keeps alive. Without this `del`, the next iteration's forward
  allocates more memory before the previous step's activations get garbage
  collected. With it, peak memory is bounded to the largest single graph.
- `del opt` — for UniGradICON-sized models Adam state is ~2× parameter size
  (~565 MB). It's wasteful to keep it alive once the pair is done.
- `torch.cuda.empty_cache()` — does **not** free memory from leaks. What it
  does is **release cached but unused blocks back to CUDA's driver**, so the
  next request (e.g. a slightly different shape on the next pair) is more
  likely to find a contiguous block. Useful in loops with varied tensor
  sizes; harmful in tight loops that allocate identical shapes (you'd just
  re-pay malloc).

### 4) Move the field extraction **inside** `torch.no_grad()`

Before:

```python
with torch.no_grad():
    net(source, target)
phi_px = phi_vectorfield_to_slice_pixels(net, orig_h, orig_w)
```

The `net(...)` does not build a graph (good). But `net.phi_AB_vectorfield`
was assigned during the forward and is **still graph-aware**. The instant
you leave the `no_grad` block, the subsequent subtraction and
`F.interpolate` re-enter grad-tracking mode and start building a fresh graph
pinned to `net.phi_AB_vectorfield`. Those graph buffers don't get freed
until the resulting Python variable falls out of scope.

After:

```python
with torch.no_grad():
    net(source, target)
    phi_px = phi_vectorfield_to_slice_pixels(net, orig_h, orig_w)
```

All post-processing happens inside `no_grad`, so nothing is graph-aware and
no activations get pinned. `phi_px` ends up as a plain numpy array detached
from PyTorch entirely.

General rule:

> `no_grad` only affects ops **inside** its block. Move every op that does
> not need a gradient inside the block.

The same applies to `torch.inference_mode()` (slightly stricter, slightly
faster, used for pure inference paths).

### 5) `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

Set in the shell, not the code:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

PyTorch's CUDA caching allocator normally manages many fixed-size **segments**
behind the scenes. After enough allocations of different sizes those segments
become Swiss cheese: total free memory is large, but no single contiguous
block is large enough for the next request. That is **fragmentation**, and
it's what causes the classic message:

> Tried to allocate 332 MiB. ... XXX MiB free. ... 1.6 GiB reserved by PyTorch
> but unallocated.

`expandable_segments:True` switches to a single, growable virtual segment per
stream that PyTorch can subdivide more freely. Net effect: many fewer
fragmentation OOMs, at the cost of slightly more allocator bookkeeping.

This is one of the highest-leverage flags in any deep-learning stack today;
it costs you nothing and is a free win in most 3D / dynamic-shape workloads.
On modern PyTorch versions it is becoming the recommended default.

### 6) Eagerly free batch-scope tensors

In the outer loop:

```python
del source
torch.cuda.empty_cache()
```

The preprocessed `source` is a `[1, 1, 175, 175, 175]` float tensor (~22 MB),
small per-sample but it persists for the lifetime of the loop variable.
Across hundreds of samples plus the post-IO `phi_px` numpy buffers, that adds
fragmentation. Cheap insurance.

## Things that did **not** turn out to matter much here

- **Adam → SGD.** SGD saves ~565 MB (Adam's momentum buffers), but the
  ~9 GiB cost is activations. Adam is also what upstream UniGradICON uses,
  so we kept it as the default after confirming the above tricks fit it on
  11 GiB.
- **AMP / fp16.** Could in principle halve activation memory, but GradICON
  uses spatial gradients and inverse-consistency terms whose numerical
  stability under fp16 has not been characterized by the authors. Not worth
  it for a per-pair IO experiment when the official protocol is fp32.

## Mental model

When you see a CUDA OOM that says **"X MiB free, Y MiB reserved but unallocated"**,
treat it as a hint that you have either:

1. **A pinned activation graph** that should have been released (fix: `no_grad`,
   `del loss`, `del optimizer`, etc.), or
2. **Fragmentation** (fix: `empty_cache`, `expandable_segments`, eager frees),
   or
3. **Genuine memory exhaustion** (fix: smaller model, smaller inputs,
   gradient checkpointing, mixed precision, a bigger GPU).

The IO script hit (1) and (2). The combination of CPU-side weight backup,
`set_to_none=True`, eager `del` + `empty_cache`, scoping post-processing
under `no_grad`, and the `expandable_segments` env var was enough to keep
the official UniGradICON IO protocol running on an 11 GiB GPU.
