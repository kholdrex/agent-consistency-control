# Third-party material

- **AQuA-RAT** (W. Ling, D. Yogatama, C. Dyer, P. Blunsom, 2017), https://github.com/google-deepmind/AQuA, commit
  26b64ed1f22208c6742f47df547dfd65088ec30d, Apache License 2.0 (LICENSES/Apache-2.0.txt), Copyright 2017 Google Inc.
  `data/corpus.json` contains the texts of the 625 training-split problems and worked solutions used as evidence passages;
  `results/` contains the indices of the test-split items and model outputs about them. The raw dataset files are
  downloaded at the pinned commit and are not redistributed here.
- **Model outputs** in `results/timing/` and `results/quality/` were produced by Gemini 3.1 Flash-Lite (Google, through
  OpenRouter) and by Qwen2.5 7B Instruct (Alibaba Cloud, Apache License 2.0) run locally with Ollama 0.34.3. They are
  published as research data under the providers' terms.
