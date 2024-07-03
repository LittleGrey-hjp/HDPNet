"""
For evaluating the performance of each method on COD10K superclasses and subclasses
Example out put:

eval-dataset: COD10K
class eval of : F:\PY_Project\DTINet/COD10K/
eval super-class: Aquatic
100%|██████████| 474/474 [01:41<00:00,  4.66it/s]
Aquatic : {'Smeasure': 0.831904818338465, 'wFmeasure': 0.7332381942337652, 'adpFm': 0.7460811157920052, 'meanFm': 0.7662376489566379, 'maxFm': 0.7908710095955639, 'adpEm': 0.8880700947495253, 'meanEm': 0.9002922741856467, 'maxEm': 0.9112305982305015, 'MAE': 0.04305738786710646}

sub-class begin
eval sub-class: BatFish
100%|██████████| 4/4 [00:00<00:00, 32.67it/s]
BatFish : {'Smeasure': 0.9021145174956894}
"""

from tqdm import tqdm
import os
import cv2
from py_sod_metrics import Emeasure, Fmeasure, Smeasure, MAE, WeightedFmeasure

method = 'HDPNet'

# 统计超类和子类
super_classes = []
sub_classes = []

for filename in os.listdir('F:\TestDataset\COD10K\GT'):
    parts = filename.split("-")
    super_class = parts[3]
    if super_class not in super_classes:
        super_classes.append(super_class)
    sub_class = parts[5]
    if sub_class not in sub_classes:
        sub_classes.append(sub_class)

result_path=['F:\PY_Project\BGNet', 'F:\PY_Project\FEDER','F:\PY_Project\SINetV2','F:\PY_Project\ZoomNet-main\output\COD_Results', 'F:\PY_Project\zoomnext_res50',
             'F:\PY_Project\DGNet',
             r'F:\PY_Project\UGTR', r'F:\PY_Project\ICON-main\results',r'F:\PY_Project\TPRNet-main\res','F:\PY_Project\DTINet',
             'F:\PY_Project\EVP\output','F:\PY_Project\FSPNet','F:\PY_Project\my_CAMO_results']

data_name = 'COD10K'
data = {}
print("eval-dataset: {}".format(data_name))
for result in result_path:
    mask_root = 'F:\TestDataset/{}/{}/'.format(data_name, "GT")  # change path
    pred_root = '{}/{}/'.format(result,data_name)  # change path
    print('\033[92m', 'class eval of :', pred_root, '\033[0m')
    pred_name_list = sorted(os.listdir(pred_root))

    for super_cls in super_classes:
        print('eval super-class:',super_cls)
        filtered_predfiles = [filename for filename in pred_name_list if super_cls in filename]
        SM = Smeasure()
        EM = Emeasure()
        WFM = WeightedFmeasure()
        FM = Fmeasure()
        M = MAE()
        for pred_name in tqdm(filtered_predfiles, total=len(filtered_predfiles)):
            mask_path = os.path.join(mask_root, pred_name)
            pred_path = os.path.join(pred_root, pred_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            FM.step(pred=pred, gt=mask)
            SM.step(pred=pred, gt=mask)
            EM.step(pred=pred, gt=mask)
            M.step(pred=pred, gt=mask)
            WFM.step(pred=pred, gt=mask)

        fm = FM.get_results()["fm"]
        sm = SM.get_results()["sm"]
        em = EM.get_results()["em"]
        mae = M.get_results()["mae"]
        wfm = WFM.get_results()["wfm"]

        results = {
            "Smeasure": sm,
            "wFmeasure": wfm,
            "adpFm": fm["adp"],
            "meanFm": fm["curve"].mean(),
            "maxFm": fm["curve"].max(),
            "adpEm": em["adp"],
            "meanEm": em["curve"].mean(),
            "maxEm": em["curve"].max(),
            "MAE": mae,
        }

        print(super_cls,':',results)

    print('\033[94m' + 'sub-class begin' + '\033[0m')
    for sub_cls in sub_classes:
        print('eval sub-class:', sub_cls)
        filtered_predfiles = [filename for filename in pred_name_list if sub_cls in filename]
        SM = Smeasure()
        for pred_name in tqdm(filtered_predfiles, total=len(filtered_predfiles)):
            mask_path = os.path.join(mask_root, pred_name)
            pred_path = os.path.join(pred_root, pred_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            SM.step(pred=pred, gt=mask)

        sm = SM.get_results()["sm"]

        results = {
            "Smeasure": sm
        }

        print(sub_cls, ':', results)
