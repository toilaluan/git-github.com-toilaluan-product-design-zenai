# [DEMO] SDXL - Sketch to Realistic Image

SDXL can generate realistic product image from sketch
In this notebook, we compare 3 SDXL variants to find the most suitable checkpoint for our goal: 
- `SDXL Default`
- `RealVis V2.0` (realistic finetuned)
- `Juggernaut XL` (realistic finetuned)

Using `TencentARC/t2i-adapter-sketch-sdxl-1.0` as controlnet.

Then, we do experiment with `SDXL Refine` to see how this model can improve generated image, it's reported that `SDXL Refine` can help to generate more realistic, textured image


## Getting started
See `main_report.ipynb` for the details
