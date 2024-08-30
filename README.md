<div align="center">
<h1>CSGO: Content-Style Composition in Text-to-Image Generation</h1>

[**Peng Xing**](https://github.com/xingp-ng)<sup>12*</sup> · [**Haofan Wang**](https://haofanwang.github.io/)<sup>1*</sup> · [**Yanpeng Sun**](https://scholar.google.com.hk/citations?user=a3FI8c4AAAAJ&hl=zh-CN&oi=ao/)<sup>2</sup> · [**Qixun Wang**](https://github.com/wangqixun)<sup>1</sup> · [**Xu Bai**](https://huggingface.co/baymin0220)<sup>1</sup> · [**Hao Ai**](https://github.com/aihao2000)<sup>13</sup> · [**Renyuan Huang**](https://github.com/DannHuang)<sup>14</sup> · [**Zechao Li**](https://zechao-li.github.io/)<sup>2✉</sup>

<sup>1</sup>InstantX Team · <sup>2</sup>Nanjing University of Science and Technology · <sup>3</sup>Beihang University · <sup>4</sup>Peking University

<sup>*</sup>equal contributions, <sup>✉</sup>corresponding authors

<a href='https://csgo-gen.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/abs/2404.02733'><img src='https://img.shields.io/badge/Technique-Report-red'></a>
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-red)](https://huggingface.co/spaces/InstantX/InstantStyle)
[![ModelScope](https://img.shields.io/badge/ModelScope-Studios-blue)](https://modelscope.cn/studios/instantx/InstantStyle/summary)
[![GitHub](https://img.shields.io/github/stars/instantX-research/CSGO?style=social)](https://github.com/instantX-research/CSGO)
</div>


##  Updates 🔥

[//]: # (- **`2024/07/19`**: ✨ We support 🎞️ portrait video editing &#40;aka v2v&#41;! More to see [here]&#40;assets/docs/changelog/2024-07-19.md&#41;.)

[//]: # (- **`2024/07/17`**: 🍎 We support macOS with Apple Silicon, modified from [jeethu]&#40;https://github.com/jeethu&#41;'s PR [#143]&#40;https://github.com/KwaiVGI/LivePortrait/pull/143&#41;.)

[//]: # (- **`2024/07/10`**: 💪 We support audio and video concatenating, driving video auto-cropping, and template making to protect privacy. More to see [here]&#40;assets/docs/changelog/2024-07-10.md&#41;.)

[//]: # (- **`2024/07/09`**: 🤗 We released the [HuggingFace Space]&#40;https://huggingface.co/spaces/KwaiVGI/liveportrait&#41;, thanks to the HF team and [Gradio]&#40;https://github.com/gradio-app/gradio&#41;!)
[//]: # (Continuous updates, stay tuned!)
[//]: # (- **`2024/08/30`**: 😊 We released the initial version of the inference code.)
- **`2024/08/30`**: 😊 We released the technical report on [arXiv](https://arxiv.org/pdf/2408.16766)
- **`2024/07/15`**: 🔥 We released the [homepage](https://csgo-gen.github.io).

##   Plan 💪
- [x]  technical report
- [ ]  inference code
- [ ]  pre-trained weight
- [ ]  IMAGStyle dataset
- [ ]  training code

## Introduction 📖
This repo, named **CSGO**, contains the official PyTorch implementation of our paper [CSGO: Content-Style Composition in Text-to-Image Generation](https://arxiv.org/pdf/).
We are actively updating and improving this repository. If you find any bugs or have suggestions, welcome to raise issues or submit pull requests (PR) 💖.

## Pipeline 	💻
<p align="center">
  <img src="assets/image3_1.jpg">
</p>

## Capabilities 🚅 

  🔥 Our CSGO achieves **image-driven style transfer, text-driven stylized synthesis, and text editing-driven stylized synthesis**.

  🔥 For more results, visit our <a href="https://csgo-gen.github.io"><strong>homepage</strong></a> 🔥

<p align="center">
  <img src="assets/vis.jpg">
</p>

## Demos

### Content-Style Composition
<p align="center">
  <img src="assets/page1.png">
</p>

<p align="center">
  <img src="assets/page4.png">
</p>

### Cycle Translation
<p align="center">
  <img src="assets/page8.png">
</p>

### Stylized Synthesis
<p align="center">
  <img src="assets/page10.png">
</p>

### Text-Driven Image Editing
<p align="center">
  <img src="assets/page11.jpg">
</p>


## Acknowledgements
This project is developed by InstantX Team, all copyright reserved.


## Citation 💖
If you find CSGO useful for your research, welcome to 🌟 this repo and cite our work using the following BibTeX:
```bibtex
@article{xing2024csgo,
       title={CSGO: Content-Style Composition in Text-to-Image Generation}, 
       author={Peng Xing and Haofan Wang and Yanpeng Sun and Qixun Wang and Xu Bai and Hao Ai and Renyuan Huang and Zechao Li},
       year={2024},
       journal = {arXiv 2408.16766},
}
```