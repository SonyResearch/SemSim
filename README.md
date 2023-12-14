# Privacy Assessment on Reconstructed Images: Are Existing Evaluation Metrics Faithful to Human Perception?

This repository is an implementation of the Semsim evalution method discussed in [“Privacy Assessment on Reconstructed Images: Are Existing Evaluation Metrics Faithful to Human Perception?”](https://arxiv.org/pdf/2309.13038.pdf)

[**Project Page**](https://sites.google.com/view/semsim)


---
The important experimental part can be found at ```benchmark/```.

The existing matrics can be found at ```metrics/```.
## Setup
You can use [anaconda](https://www.anaconda.com/distribution/) to install our setup by running
```
conda env create -f semsim.yml
conda activate semsim
```


## Getting Started
####  Step1  train classifier to be evaluatred
```
# using cifar-100 as example
python benchmark/step1_train_classifier.py --data=cifar100 --arch=ResNet20-4 --epochs=200 --aug_list='' --mode=crop
```


####  Step2 attack classifier to get reconstructed images
```
# using cifar-100 as example
python benchmark/step2_attack.py --data=cifar100 --arch=ResNet20-4 --epochs=200 --aug_list='' --mode=crop --optim='inversed
```


#### Step3 use different metric to measure the privacy leakage
```
# exisitng metric

python metrics/pixel_level_metrics

# modify line 137-138 
#    data_dir_raw= '' # dir of orginal images 
#    with open('metrics/folder_names_cifar.txt', 'r') as f: 
#    folder_names_cifar.txt saves dirs of reconstructed images 
```


```
# train Semsim
python benchmark\Semsim_train_evaluation.py --data human_anno_id --arch ResNet18 --epochs 100 --mode crop --semsim True
```


```
# test Semsim
python benchmark\Semsim_train_evaluation.py --data human_anno_id --arch ResNet18 --epochs 100 --mode crop --semsim True --evaluate True
```

#### Step4 analyse results

```
# caculate correlation between different metrics
python metrics\models_rank_corr.py 
```

# Citation 

Please cite this paper if it helps your research:
```bibtex
@inproceedings{sun2023privacy,
  title={Privacy Assessment on Reconstructed Images: Are Existing Evaluation Metrics Faithful to Human Perception?},
  author={Sun, Xiaoxiao and Gazagnadou, Nidham and Sharma, Vivek and Lyu, Lingjuan and Li, Hongdong and Zheng, Liang},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```

# Acknowledgement 
We express gratitudes to the great work [ATSPrivacy](https://github.com/gaow0007/ATSPrivacy), [Inverting Gradients](https://github.com/JonasGeiping/invertinggradients) and [DLG](https://github.com/mit-han-lab/dlg) as we benefit a lot from both their papers and codes.

# License
This repository is released under the MIT license. 
