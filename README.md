# Isometric Representation Learning for Disentangled Latent Space of Diffusion Models
> Jaehoon Hahm\*, Junho Lee\*, Sunghyun Kim, Joonseok Lee
> (* equal contribution)

![ICML 2024](https://img.shields.io/badge/ICML-2024-blue)

https://arxiv.org/abs/2407.11451

This repository is the official Pytorch implementation for Isometric diffusion.

![teaser](assets/figures/figure1.png)
<p align="justify">
    <strong>An illustration of latent traversal between two latents $x$ and $x'$</strong>.
    <em>Top</em>: naive linear interpolation (Lerp) assuming Euclidean space. The starting point $x_T$ is linearly interpolated.
    <em>Mid</em>: spherical interpolation (Slerp) between $x$ and $x'$ (direction $x \rightarrow x'$ is entangled with an unwanted gender axis, inducing abrupt changes).
    <em>Bottom</em>: Slerp with the same latents using our Isometric Diffusion to resolve unwanted entanglement.
</p>

## Setup
**Setting the environment.**
```
conda create --name isodiff python=3.9
conda activate isodiff
pip install torch==2.0.1 torchvision==0.15.2
pip install -r requirements.txt
```

**Setting the Dataset.**
Change the 'DATASET_PATH' in 'submit_celeba.sh' and 'submit_celeba_ldm.sh' to CelebA-HQ dataset path.
Your directory structure should look like:
```
$DATASET_PATH/xxx.png
$DATASET_PATH/xxy.png
$DATASET_PATH/[...]/xxz.png
```

## Training
```
# set accelerate. [https://huggingface.co/docs/accelerate/quicktour] 
# DDPM
bash submit_celeba.sh

# LDM
bash submit_celeba_ldm.sh
```
You can find the pre-trained weights for the Isometric Diffusion model of CelebA-HQ at this [link](https://drive.google.com/drive/folders/141YEtNYwk6YRoVHhpMFP7D6Po9K7VhOB?usp=drive_link).

## Results
### Image Interpolation
![teaser](assets/figures/figure5.png)

### Image Inversion and Reconstruction
<p align="center">
  <img src="assets/figures/figure6.png" alt="teaser" width="500"/>
</p>

### Linearity
![teaser](assets/figures/figure7.png)
### Linear Combination
![teaser](assets/figures/figure8.png)

## Citation
```

@article{hahm2024isometric,
  title={Isometric Representation Learning for Disentangled Latent Space of Diffusion Models},
  author={Hahm, Jaehoon and Lee, Junho and Kim, Sunghyun and Lee, Joonseok},
  journal={arXiv preprint arXiv:2407.11451},
  year={2024}
}

```
```
@InProceedings{pmlr-v235-hahm24a,
  title = 	 {Isometric Representation Learning for Disentangled Latent Space of Diffusion Models},
  author =       {Hahm, Jaehoon and Lee, Junho and Kim, Sunghyun and Lee, Joonseok},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {17224--17245},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/hahm24a/hahm24a.pdf},
  url = 	 {https://proceedings.mlr.press/v235/hahm24a.html},
  abstract = 	 {The latent space of diffusion model mostly still remains unexplored, despite its great success and potential in the field of generative modeling. In fact, the latent space of existing diffusion models are entangled, with a distorted mapping from its latent space to image space. To tackle this problem, we present Isometric Diffusion, equipping a diffusion model with a geometric regularizer to guide the model to learn a geometrically sound latent space of the training data manifold. This approach allows diffusion models to learn a more disentangled latent space, which enables smoother interpolation, more accurate inversion, and more precise control over attributes directly in the latent space. Our extensive experiments consisting of image interpolations, image inversions, and linear editing show the effectiveness of our method.}
}

```
