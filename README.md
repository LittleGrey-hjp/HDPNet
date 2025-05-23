# HDPNet: Hourglass Vision Transformer with Dual-Path Feature Pyramid for Camouflaged Object Detection (WACV2025)

## Usage

> The training and testing experiments are conducted using PyTorch with an NVIDIA A100-SXM of 40 GB Memory.

### 1. Prerequisites

> Note that HDPNet is only tested on Ubuntu OS with the following environments.

- Creating a virtual environment in terminal: `conda create -n HDPNet python=3.8`.
- Installing necessary packages: `pip install -r requirements.txt`

### 2. Downloading Training and Testing Datasets

- Download the [training set](https://drive.google.com/drive/folders/1V1z3WxDOqZeo8adIu6vmh8rZPf31M_To?usp=drive_link) (COD-TrainDataset) used for training 
- Download the [testing sets](https://drive.google.com/drive/folders/1UndxEPkZFxZwbrVvctb8ZvRpThwB7AND?usp=drive_link) (COD10K-test + CAMO-test + CHAMELEON + NC4K ) used for testing

### 3. Training Configuration

- The pretrained model(PVT2) is stored in [Google Drive](https://drive.google.com/file/d/1fJpCAKDIISC5yQcr4XASalv95hdI5cB4/view?usp=drive_link) and [Baidu Drive](https://pan.baidu.com/s/1WhKe3unTSfsHboCzUu0OqQ) (g3ea). After downloading, please change the file path in the corresponding code.
- Run `train.sh` to train.

### 4. Testing Configuration

Our well-trained model is stored in [Google Drive](https://drive.google.com/file/d/1LfKhIV0cXl_lNpkrLsv4TMvcRMx_IIYW/view?usp=drive_link) and [Baidu Drive](https://pan.baidu.com/s/1ESnWJ19ivgSrOxadWJRDuQ) (gv9n). After downloading, please change the file path in the corresponding code.

### 5. Evaluation

- Evaluate HDPNet: After configuring the test dataset path, run `hpvt_eval.sh` in the `run_slurm` folder for evaluation.
- PR-Curves: We provide the code for obtaining PR-Curves through detection results. Please refer to 'PR_Curve.py'.
- Super- and Sub-Classes: To evaluate the performance of each method on COD10K superclasses and subclasses through detection results, please refer to 'class_eval.py'.

### 6. Results download

The prediction results of our HDPNet are stored on [Google Drive](https://drive.google.com/drive/folders/1znoCKopi-CtAxj2-ixOTjjx833-PlxAx?usp=drive_link). Please check.

### 7. Quantitative Results
> 
> Our final results,  which perform very well on the COD10K dataset (contains a lot of small objects and detailed labeling of the objects' fine boundaries).
>
> we adopt five kinds of evaluation metrics:
> S-measure($S_m$), weighted F-measure($F_\beta^\omega$), adaptive F-measure($F_\beta^{adp}$), mean F-measure($F_\beta^{mean}$),max F-measure($F_\beta^{max}$), adaptive E-measure($E_\phi^{apd}$),
> mean E-measure($E_\phi^{mean}$), max E-measure ($E_\phi^{max}$), and mean absolute error($\mathcal{M}$)
> 
| Dataset   | $S_m \uparrow$ | $F_\beta^\omega \uparrow$ | $F_\beta^{adp} \uparrow$ | $F_\beta^{mean} \uparrow$ | $F_\beta^{max} \uparrow$ | $E_\phi^{adp} \uparrow$ | $E_\phi^{mean} \uparrow$ | $E_\phi^{max} \uparrow$ | $\mathcal{M} \downarrow$ |
|:---------:|:--------------:|:--------------------:|:----------------:|:----------------:|:----------------:|:----------------:|:----------------:|:----------------:|:-------------------------:|
| CAMO      |    0.893       |        0.851         |      0.848       |      0.870       |      0.890       |      0.932       |      0.934       |      0.948       |           0.040           |
| CHAMELEON |    0.921       |        0.861         |      0.849       |      0.874       |      0.902       |      0.943       |      0.947       |      0.970       |           0.021           |
| COD10K    |    0.888       |        0.794         |      0.770       |      0.820       |      0.852       |      0.915       |      0.925       |      0.951       |           0.020           |
| NC4K      |    0.902       |        0.850         |      0.845       |      0.871       |      0.891       |      0.931       |      0.934       |      0.950       |           0.029           |
> 
### 8. Qualitative Results
>
>Quantitative results in several typical complex situations, including occlusion, small objects, multiple objects, and object boundaries.
> 
![Qualitative Result](https://github.com/LittleGrey-hjp/HDPNet/blob/main/Visio-camouflage_fig1.jpg)


## Citation

```
@inproceedings{he2025hdpnet,
  title={HDPNet: Hourglass Vision Transformer with Dual-Path Feature Pyramid for Camouflaged Object Detection},
  author={He, Jinpeng and Liu, Biyuan and Chen, Huaixin},
  booktitle={2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={8638--8647},
  year={2025},
  organization={IEEE}
}
```
