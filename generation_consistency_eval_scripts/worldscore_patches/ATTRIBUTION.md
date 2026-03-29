# Attribution

The generation consistency evaluation in WR-Arena builds on
[WorldScore](https://github.com/haoyi-duan/WorldScore), which is licensed
under the MIT License.

```
MIT License

Copyright (c) 2025 Haoyi Duan

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

## What we changed

`evaluator_per_round_arif.py` is derived from WorldScore's evaluator class.
Changes relative to the original:
- Removed `MultiRoundSmoothness_v1Metric` (VFIMamba-based temporal smoothness)
  and all associated code paths so the evaluator depends only on standard
  WorldScore metrics.
- `model_type` registry lookup uses `.get()` with a fallback so WR-Arena model
  names that are not in WorldScore's built-in registry are handled gracefully.

## Citation

If you use this evaluation in your work, please cite WorldScore:

```bibtex
@article{duan2025worldscore,
  title   = {WorldScore: A Unified Evaluation Benchmark for World Generation},
  author  = {Haoyi Duan and others},
  year    = {2025},
}
```
