# IR Diffusion Playground

Scratchpad for low-cost (≤16 GB VRAM) IR/diffusion prototypes:

- Diffusion-as-corrector: draft IR → denoise → speaker; keep air-gap tests (scramble IR should break).
- Diffusion-as-search: sample K IR variants → score/pick best (simple scorer or answer logprob).

Use tiny models/datasets (e.g., small arithmetic/story), track math exact-match, story token F1, IR usage/entropy, and IR scramble effects before scaling.***
