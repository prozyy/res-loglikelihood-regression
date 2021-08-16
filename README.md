# Human Pose Regression with Residual Log-likelihood Estimation

[[`Paper`](https://jeffli.site/res-loglikelihood-regression/resources/ICCV21-RLE.pdf)]
[[`arXiv`](https://arxiv.org/abs/2107.11291)]
[[`Project Page`](https://jeffli.site/res-loglikelihood-regression/)]

> [Human Pose Regression with Residual Log-likelihood Estimation](https://jeffli.site/res-loglikelihood-regression/resources/ICCV21-RLE.pdf)  
> Jiefeng Li, Siyuan Bian, Ailing Zeng, Can Wang, Bo Pang, Wentao Liu, Cewu Lu  
> ICCV 2021 Oral  

<div align="center">
    <img src="assets/rle.jpg", width="600" alt><br>
    Regression with Residual Log-likelihood Estimation
</div>

## TODO
- [ ] Provide minimal implementation of RLE loss.
- [ ] Add model zoo.
- [x] Provide implementation on Human3.6M dataset.
- [x] Provide implementation on COCO dataset.

### Installation
1. Install pytorch >= 1.1.0 following official instruction.
2. Install `rlepose`:
``` bash
pip install cython
python setup.py develop
```
3. Install [COCOAPI](https://github.com/cocodataset/cocoapi).
``` bash
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
4. Init `data` directory:
``` bash
mkdir data
```
5. Download [COCO](https://cocodataset.org/#download) data and [Human3.6M](https://drive.google.com/drive/folders/1kgVH-GugrLoc9XyvP6nRoaFpw3TmM5xK) data (from [PoseNet](https://github.com/mks0601/3DMPPE_POSENET_RELEASE) or [ours](https://drive.google.com/file/d/1jh_nnQo2wMFCffGCn8Xh3xvlCVgBuWPZ/view?usp=sharing)):
```
|-- data
`-- |-- coco
    |   |-- annotations
    |   |   |-- person_keypoints_train2017.json
    |   |   `-- person_keypoints_val2017.json
    |   `-- images
    |       |-- train2017
    |       |   |-- 000000000009.jpg
    |       |   |-- 000000000025.jpg
    |       |   |-- 000000000030.jpg
    |       |   |-- ... 
    |       `-- val2017
    |           |-- 000000000139.jpg
    |           |-- 000000000285.jpg
    |           |-- 000000000632.jpg
    |           |-- ... 
    |-- h36m
    `-- |-- annotations
        |   |-- Sample_trainmin_train_Human36M_protocol_2.json
        |   `-- Sample_64_test_Human36M_protocol_2.json
        `-- images
            |-- s_01_act_02_subact_01_ca_01
            |   |-- ...
            |-- s_01_act_02_subact_01_ca_02
            |   |-- ...
            `-- ... 
```
## Training

### Train on MSCOCO
``` bash
./scripts/train.sh ./configs/256x192_res50_regress-flow.yaml train_rle_coco
./scripts/train.sh ./configs/256x192_hrnetw32_regress-flow.yaml train_rle
```

### Train on Human3.6M
``` bash
./scripts/train.sh ./configs/256x192_res50_3d_h36mmpii-flow.yaml train_rle_h36m
```

## Evaluation

### Validate on MSCOCO
Download the pretrained model from [Google Drive](https://drive.google.com/file/d/1YBHqNKkxIVv8CqgDxkezC-4vyKpx-zXK/view?usp=sharing).
``` bash
./scripts/validate.sh ./configs/256x192_res50_regress-flow.yaml ./coco-laplace-rle.pth
```
### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset(test_det_rcnn.json)
| Arch               | Input size | #Params | GFLOPs |    AP | Ap .5 | AP .75 | AP (M) | AP (L) |    AR | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------------|------------|---------|--------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| rle_resnet_50      |    256x192 |  23.6M  |  3.73  | 0.693 | 0.878 |  0.764 |  0.666 |  0.748 | 0.744 | 0.915 |  0.809 |  0.706 |  0.802 |
| rle_resnet_50*     |    256x192 |  23.6M  |  3.73  | 0.709 | 0.889 |  0.776 |  0.677 |  0.769 | 0.760 | 0.925 |  0.819 |  0.719 |  0.819 |
| rle_hrnet_w32      |    256x192 |  39.3M  |  8.14  | 0.720 | 0.879 |  0.785 |  0.694 |  0.773 | 0.768 | 0.915 |  0.826 |  0.732 |  0.821 |
| rle_hrnet_w32*     |    256x192 |  39.3M  |  8.14  | 0.750 | 0.896 |  0.816 |  0.716 |  0.810 | 0.798 | 0.933 |  0.857 |  0.757 |  0.858 |
| pose_hrnet_w32     |    256x192 |  28.5M  |  7.1   | 0.744 | 0.905 |  0.819 |  0.708 |  0.810 | 0.798 | 0.942 |  0.865 |  0.757 |  0.858 |

### Note:
- Flip test is used.
- *means pretained Model get from  train Heatmap based method

### Validate on Human3.6M
Download the pretrained model from [Google Drive](https://drive.google.com/file/d/1v2ZhembnFyJ_FXGHEOCzGaM-tAVFMy7A/view?usp=sharing).
``` bash
./scripts/validate.sh ./configs/256x192_res50_3d_h36mmpii-flow.yaml ./h36m-laplace-rle.pth
# PA-MPJPE 38.481315, Protocol 2 error (MPJPE) >> tot: 48.756247
```

### Citing
If our code helps your research, please consider citing the following paper:
```
@inproceedings{li2021human,
    title={Human Pose Regression with Residual Log-likelihood Estimation},
    author={Li, Jiefeng and Bian, Siyuan and Zeng, Ailing and Wang, Can and Pang, Bo and Liu, Wentao and Lu, Cewu},
    booktitle={ICCV},
    year={2021}
}
```