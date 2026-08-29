# Third-Party Notices

This repository is licensed under Apache-2.0 (see `LICENSE`), with the
following third-party components and datasets, credited here with their
original licenses.

## nanoGPT (MIT)

The transformer implementations in `air_gap/*/model.py` (GPTConfig /
CausalSelfAttention / Block structure) are adapted from Andrej Karpathy's
nanoGPT: https://github.com/karpathy/nanoGPT

```
MIT License

Copyright (c) 2022 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## GSM8K (MIT)

Some result files under `vq_bottleneck/results*/` embed evaluation samples
(questions and answers) from OpenAI's GSM8K dataset:
https://github.com/openai/grade-school-math

```
MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## bAbI tasks (CC BY 3.0)

`air_gap/v13/tasks_1-20_v1-2/` contains the bAbI tasks dataset (Facebook AI
Research), distributed under Creative Commons Attribution 3.0; the original
`LICENSE.txt` and `README.txt` are preserved in that folder.

## Datasets referenced but not included

- **TinyStories** (Eldan & Li): used for training in `air_gap/` runs;
  obtain it from its original distribution under its own license.
- **Pythia** model weights (EleutherAI, Apache-2.0): used as backbones in
  `vq_bottleneck/` and `seed_emergent_ir/`; weights are not included.
