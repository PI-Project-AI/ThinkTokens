# ThinkTokens: Project Structure (current)

**Goal:** Research on machine-native Internal Reasoning (IR) via discrete tokens and air-gapped architectures.

**Approach:** Separate, self-contained experiment folders per architecture/version.

---

## Project Organization

```
ThinkTokens/
├── air_gap/              ← Air-gap lineage (v10 → v18, canonical)
│   ├── v10/ ... v18/                ← Each version is standalone
│   ├── ir_diffusion_playground/     ← Early IR diffusion prototypes
│   └── v18/h100_snapshot/v18_full/  ← H100 export snapshot (archived, non-exploitable as-is)
│
├── seed_emergent_ir/                ← Seed + emergent IR architecture (separate line)
├── vq_bottleneck/                   ← VQ bottleneck experiments (archived baseline)
├── two_model_architecture/          ← Two-model / LA pivot work
├── hybrid_parallel_reasoning/       ← IR + CoT parallel reasoning (concept)
├── legacy_root/                     ← Archived root scripts and docs
│
├── docs/                            ← Documentation (current + archive)
│   ├── TiDAR/                       ← Diffusion-related notes
│   ├── eSoleau_INPI/                ← Filing materials
│   └── archive/                     ← Older guides and plans
│
├── README.md                        ← Public release overview
├── LICENSE / NOTICE / CITATION.cff
└── requirements.txt
```

---

## Organization Rules

- One top-level folder per architecture line (air-gap, seed+emergent, two-model, etc.).
- Each version folder is standalone and owns its results/checkpoints.
- Archived exports live under the owning architecture (e.g., `air_gap/v18/h100_snapshot/`).
- V18 H100 export is a forensic snapshot, not a valid final result: Phase 2 collapsed to `<unk>` and usable eval logs were not produced.
- `docs/archive/` contains historical documents that are not kept in sync.
