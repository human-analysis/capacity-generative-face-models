# On the Biometric Capacity of Generative Face Models [[arXiv]](https://arxiv.org/abs/arXiv:2308.02065)

```BibTex
@inproceedings{boddeti2023biometric,
  title={On the Biometric Capacity of Generative Face Models},
  author={Boddeti, Vishnu Naresh and Sreekumar, Gautam and Ross, Arun},
  booktitle={International Joint Conference on Biometrics (IJCB)},
  year={2023}
}
```

## Overview

This work presents the **first** approach for estimating the *biometric capacity* of a given generative face model. Capacity is the *maximal number of unique identities that a given generative face model can generate.* 

![capacity estimation concept for hyper-spherical representation space](./assets/overview.png "capacity estimation concept for hyper-spherical representation space")

### Capacity Estimation Concept

The idea is to represent the generated face images in a hyperspherical space, i.e., $\|z\|=1$, and estimate capacity as a ratio of hyper-spherical caps corresponding to all classes (inter-class variance) and a single class (intra-class variance).

We estimate capacity as a ratio of hyper-spherical caps corresponding to all classes (inter-class variance) and a single class (intra-class variance).

![alt text](./assets/capacity-estimation-concept.png "capacity estimation concept for hyper-spherical representation space")

For the general case of a $n$-dimensional representation space, the ratio of hyper-spherical caps with solid angles $\Omega_1$ and $\Omega_2$ is,

$$C(\Omega_1, \Omega_2) = \frac{I_{sin^2(\Omega_1)}\left(\frac{n-1}{2},\frac{1}{2}\right)}{I_{sin^2(\Omega_2)}\left(\frac{n-1}{2},\frac{1}{2}\right)}$$

The key challenge here is accurately estimating the inter-class ($\Omega_1$) and intra-class ($\Omega_2$) angles.

## Assumptions

Before we proceed, we list the assumptions made by our proposed solution to estimate capacity.

- We estimate capacity within a feature space such as ArcFace and AdaFace. The capacity estimate is for a combination of a generative model and a feature extractor. However, this is a well-justified choice. First, raw image pixels entangle identity and geometric and photometric variations. Moreover, since we aim to estimate capacity w.r.t. unique identities instead of unique images, we need to calculate capacity in a representation space that preserves identity while being invariant to other factors. Thus, a face recognition system’s feature space is a well-justified representation choice.

- We estimate the inter-class and intra-class solid angle support from the furthest distance between all the respective samples. So, we inherently assume that the generated samples span the whole representation space within the estimated support. In practice, there may be regions of low feature density, which our approach ignores. Nonetheless, our capacity estimates are upper bounds of the actual capacity and thus still valuable to practitioners and researchers alike.

- We use a single estimate of the intra-class variance to compute capacity. However, classes typically differ in their intra-class variance due to inherent class properties or the number of samples per class.

## How to use

```bash
git clone https://github.com/human-analysis/capacity-generative-face-models
cd capacity-generative-face-models
```

### 1. Download the pre-extracted features from multiple generative face models.

​	Create a folder named features and download the features.

```bash
mkdir features
```

- For ArcFace
  ```bash
  # PGAN
  
  mkdir features/arcface
  
  wget -O features/arcface/pggan_celebahq_1024.pkl https://www.dropbox.com/scl/fi/v7zr1mjixna42024tzupv/pggan_celebahq_1024.pkl?rlkey=x0yylm41c0y6xxu3hh4n7ggt5&dl=1
  
  # StyleGAN2-Ensemble
  wget -O features/arcface/stylegan2-ensem.pkl https://www.dropbox.com/scl/fi/96785gs6txsj0zwc62xh7/stylegan2-ensem.pkl?rlkey=b30fjn5oy4da1iyec1aamllvz&dl=1
  
  # StyleGAN3
  wget -O features/arcface/stylegan3.pkl https://www.dropbox.com/scl/fi/k0k76pigzbnat49ktexxf/stylegan3.pkl?rlkey=jsg4s0uxhj15jkovfaxf9mkq5&dl=1
  
  # Latent Diffusion
  wget -O features/arcface/ldm_celebahq_256.pkl https://www.dropbox.com/scl/fi/qmfz69zo9wiee50yh5r5g/ldm_celebahq_256.pkl?rlkey=58zb255x375k8pif8obec0kzm&dl=1
  
  # Generated.Photos
  wget -O features/arcface/generated.photos.pkl https://www.dropbox.com/scl/fi/7eiza4e0hxx0rdzewgrbo/generated.photos.pkl?rlkey=lrvik48cjvihnu6k2nr54cxhw&dl=1
  
  # DCFace
  wget -O features/arcface/dcface_0.5m.pkl https://www.dropbox.com/scl/fi/twe5yjyj5r0zcahg9cy6y/dcface_0.5m.pkl?rlkey=awl795v2lzhorf2qhxdtroxar&dl=1
  ```
  
- For AdaFace
  ```bash
  # PGGAN
  
  mkdir features/adaface
  
  wget -O features/adaface/pggan_celebahq_1024.pkl https://www.dropbox.com/scl/fi/x2nrg2stu6w7dckyed400/pggan_celebahq_1024.pkl?rlkey=xanxngi06jt7rqhjqks10r73x&dl=1
  
  # StyleGAN2-Ensemble
  wget -O features/adaface/stylegan2-ensem.pkl https://www.dropbox.com/scl/fi/7gva422lcp2wduzv079ma/stylegan2-ensem.pkl?rlkey=psgzw3hkv2e7fneo82pp69g98&dl=1
  
  # StyleGAN3
  wget -O features/adaface/stylegan3.pkl https://www.dropbox.com/scl/fi/9mz8vtnm18d2suzkvudtb/stylegan3.pkl?rlkey=clmnl035o4e24cwa1y1et4trw&dl=1
  
  # Latent Diffusion
  wget -O features/adaface/ldm_celebahq_256.pkl https://www.dropbox.com/scl/fi/g0sj9x0y7ciwky5zg1tln/ldm_celebahq_256.pkl?rlkey=j42npo4godbm7b61h2umcwkcl&dl=1
  
  # Generated.Photos
  wget -O features/adaface/generated.photos.pkl https://www.dropbox.com/scl/fi/nkk4zk74n2th97xnc1gbd/generated.photos.pkl?rlkey=8veot2a9gia79eo6vuakbu5s9&dl=1
  
  # DCFace
  wget -O features/adaface/dcface_0.5m.pkl https://www.dropbox.com/scl/fi/dgkajfrqxbvtad355fqwj/dcface_0.5m.pkl?rlkey=gwcsr4u221d8zgkwk5tcwxs23&dl=1
  ```
  
  

### 2. Run the demo notebook

Download the ARCFace features from the links above for StyleGAN3 and DCFace. The demo notebook ```demo.ipynb``` walks you through the capacity estimation process for an unconditional (StyleGAN3) and a class-conditional (DCFace) generator.

### 3. Run all capacity estimation experiments

To replicate all results from the paper,

1. Download all the extracted features from the links above.
2. Modify paths, generative model names, feature extractor names etc. in ```constants.py``` if necessary.
3. run ```python3 main.py```

## Requirements

A `requirements.txt` file has been provided.
