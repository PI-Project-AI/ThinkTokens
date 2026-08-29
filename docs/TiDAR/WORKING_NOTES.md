# IR-Diffusion Working Notes (in-progress)

## Current obstacle
- Core ideas (machine-native IR, air-gap, two-phase) are solid, but the **architecture/training is brittle**: lots of arbitrary glue (block/IR split, codebook, batch/LR, eval) makes IR hard to learn/use causally.
- V18 will show how far the current air-gap + VQ stack can go at scale; risk is that brittleness persists.

## Low-cost probes (<=16 GB VRAM)
1) **Diffusion-as-corrector**  
   - Pipeline: draft IR → diffusion denoise → speaker.  
   - Goal: make the IR channel more forgiving while keeping air-gap tests (scramble IR still breaks).  
   - Toy task: small arithmetic/story synthetic; batch modest.

2) **Diffusion-as-search**  
   - Pipeline: draft IR → sample K denoised variants → score/pick best (simple scorer or answer logprob).  
   - Goal: treat IR as a state space to explore instead of forcing one perfect IR in one pass.

Both stay aligned with the pillars but add levers beyond a single VQ bottleneck.

## Notes
- Start local (no H100): small model, small IR (e.g., 16–32 slots, 256–512 codes), simple dataset.  
- Metrics: math exact-match, story token F1; IR usage/entropy; effect of IR scramble.  
- If promising, fold into a scaled run; if not, we fall back to pure AR baseline next.***
