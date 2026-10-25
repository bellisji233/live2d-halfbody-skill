"""See-through on this machine: inference_psd.py with what an Apple GPU (MPS)
needs. Run by `rigkit.seethrough fetch --site local` with See-through's own
Python, in its folder, after vendor/see-through-mac.patch; it imports
nothing from rigkit.

  python seethrough_local.py --srcp IMAGE --save_dir OUT [--resolution 1280] [--head-box X,Y,W,H]

Without --head-box: both passes, depth, and the PSD (OUT/<image name>.psd).
With it (an expression variant of an image already run): the head pass
only, on that box (px at --resolution), and the head layers as PNGs in
OUT/<image name>/.

What MPS needs, each the same maths:
- attention in chunks: there is no memory-efficient kernel, and at 1536
  with 13 layers at once the attention matrix is one 20 GB buffer;
- group_norm in chunks: in bf16 it goes through a float32 decomposition
  with full-size temporaries (1.2 GB for 13 layers at 1280);
- the tag embeddings in float32: in bf16 the CLIP text encoders get the
  first prompt of every batch wrong, and that layer comes out empty.
"""
import os
import torch
import torch.nn.functional as F

BUDGET = int(os.environ.get('SDPA_BUDGET_MB', '1024')) << 20
GN_LIMIT = 256 << 20
_sdpa, _group_norm = F.scaled_dot_product_attention, F.group_norm


def chunked_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    if q.device.type != 'mps' or q.dim() != 4 or is_causal or kw:
        return _sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
    B, H, Nq, D = q.shape
    Nk = k.shape[-2]
    row = Nk*q.element_size()*2                       # scores and softmax, one query row
    if B*H*Nq*row <= BUDGET:
        return _sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, scale=scale)
    q3, k3, v3 = (t.reshape(B*H, t.shape[-2], t.shape[-1]) for t in (q, k, v))
    m3 = None if attn_mask is None else attn_mask.expand(B, H, Nq, Nk).reshape(B*H, Nq, Nk)
    out = torch.empty(B*H, Nq, v.shape[-1], dtype=q.dtype, device=q.device)
    qs = max(1, min(Nq, BUDGET//row))                 # query rows per chunk
    bh = max(1, BUDGET//(qs*row))                     # batch*heads per chunk
    for i in range(0, B*H, bh):
        for j in range(0, Nq, qs):
            m = None if m3 is None else m3[i:i+bh, j:j+qs]
            out[i:i+bh, j:j+qs] = _sdpa(q3[i:i+bh, j:j+qs], k3[i:i+bh], v3[i:i+bh],
                                        attn_mask=m, dropout_p=dropout_p, scale=scale)
    return out.reshape(B, H, Nq, -1)


def chunked_group_norm(x, num_groups, weight=None, bias=None, eps=1e-5):
    """Each sample and each group is normalised on its own, so splitting by
    sample, then by whole groups, gives the same result."""
    if x.device.type != 'mps' or x.numel()*4 < GN_LIMIT:
        return _group_norm(x, num_groups, weight, bias, eps)
    if x.shape[0] > 1:
        return torch.cat([chunked_group_norm(c, num_groups, weight, bias, eps) for c in x.split(1)], 0)
    cpg = x.shape[1]//num_groups                      # channels per group
    k = max(1, GN_LIMIT//(x[:, :cpg].numel()*4))      # groups per chunk
    if k >= num_groups:
        return _group_norm(x, num_groups, weight, bias, eps)
    out = []
    for g in range(0, num_groups, k):
        c0, c1 = g*cpg, min(num_groups, g+k)*cpg
        out.append(_group_norm(x[:, c0:c1], (c1-c0)//cpg, None if weight is None else weight[c0:c1],
                               None if bias is None else bias[c0:c1], eps))
    return torch.cat(out, 1)


F.scaled_dot_product_attention = chunked_sdpa
F.group_norm = chunked_group_norm


def patch_pipeline(head_box=None):
    """Tag embeddings in float32 on MPS, and the body pass.
    The head pass crops the body pass's head layer where its alpha > 15. A few
    faint pixels on the image border (7 on one character) stretch that box to
    the whole image, and the head pass then runs on the whole image instead of
    the head enlarged: specks under 1% of the head go (the layer is only read
    for its box). head_box: no body pass; a head layer that fills this box."""
    import numpy as np
    import cv2
    from types import SimpleNamespace
    from modules.layerdiffuse.diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline as KD
    enc, call = KD.encode_cropped_prompt_77tokens, KD.__call__

    def encode_fp32(self, prompt):
        dt = self.text_encoder.dtype
        if self.text_encoder.device.type != 'mps' or dt == torch.float32:
            return enc(self, prompt)
        for m in (self.text_encoder, self.text_encoder_2):
            m.to(torch.float32)
        try:
            e, pooled = enc(self, prompt)
        finally:
            for m in (self.text_encoder, self.text_encoder_2):
                m.to(dt)
        return e, pooled.to(dt)

    def body_pass(self, *args, **kw):
        if kw.get('group_index') != 0:
            return call(self, *args, **kw)
        tags = list(kw['prompt'])
        if head_box:
            h, w = kw['fullpage'].shape[:2]
            images = [np.zeros((h, w, 4), np.uint8) for _ in tags]
            x, y, bw, bh = head_box
            images[tags.index('head')][y:y+bh, x:x+bw] = 255
            return SimpleNamespace(images=images)
        images = list(call(self, *args, **kw).images)
        if 'head' in tags:
            head = images[tags.index('head')].copy()
            n, lab, st, _ = cv2.connectedComponentsWithStats((head[..., 3] > 15).astype(np.uint8))
            if n > 2:
                big = st[1:, cv2.CC_STAT_AREA].max()
                head[..., 3][np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] < .01*big])] = 0
            images[tags.index('head')] = head
        return SimpleNamespace(images=images)

    KD.encode_cropped_prompt_77tokens = encode_fp32
    KD.__call__ = body_pass


def free_memory(inference_utils):
    """Unified memory: the layer model goes before the depth model loads."""
    import gc
    inference_utils.layerdiff_pipeline = None
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    import argparse, time
    from utils import inference_utils
    from utils.inference_utils import apply_layerdiff, apply_marigold, further_extr
    from utils.torch_utils import seed_everything
    ap = argparse.ArgumentParser()
    ap.add_argument('--srcp', required=True)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--resolution', type=int, default=1280)
    ap.add_argument('--inference_steps', type=int, default=30)
    ap.add_argument('--head-box', default=None, help='X,Y,W,H at --resolution: head pass only')
    a = ap.parse_args()
    box = [int(v) for v in a.head_box.split(',')] if a.head_box else None
    patch_pipeline(box)
    t0 = time.time()
    seed_everything(a.seed)
    apply_layerdiff(a.srcp, 'layerdifforg/seethroughv0.0.2_layerdiff3d', save_dir=a.save_dir, seed=a.seed,
                    resolution=a.resolution, num_inference_steps=a.inference_steps)
    print(f'layers {time.time()-t0:.0f} s', flush=True)
    if not box:
        free_memory(inference_utils)
        apply_marigold(a.srcp, '24yearsold/seethroughv0.0.1_marigold', save_dir=a.save_dir, seed=a.seed, resolution=768)
        saved = os.path.join(a.save_dir, os.path.splitext(os.path.basename(a.srcp))[0])
        further_extr(saved, rotate=False, save_to_psd=True, tblr_split=True)
        print(f'all {time.time()-t0:.0f} s', flush=True)
