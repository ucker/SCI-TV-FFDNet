# SCI Reconstruction with TV and FFDNet
Code for **Effective Snapshot Compressive-spectral Imaging via Deep Denosing and Total Variation Priors** (CVPR 2021). 

## Requirements
- python3
- pytorch, cvxpy, scipy, numpy, skimage, tqdm

## Usage
- run our algorithm
```python
python pnp_gap_HSI_tv_ffdnet.py
```

- run two baseline algorithms
```python
# Run plug-and-play gap based on 3d TV denoiser
python pnp_gap_HSI_3dtv.py
# Run plug-and-play gap based on FFDNet
python pnp_gap_HSI_ffdnet.py
```
