# Privacy Assessment on Reconstructed Images: Are Existing Evaluation Metrics Faithful to Human Perception?

<<<<<<< Updated upstream
This repository is an implementation of the SemSim evalution method discussed in [“Privacy Assessment on Reconstructed Images: Are Existing Evaluation Metrics Faithful to Human Perception?”](https://arxiv.org/pdf/2309.13038.pdf)
=======
This repository is an implementation of the Semsim evaluation method discussed in [“Privacy Assessment on Reconstructed Images: Are Existing Evaluation Metrics Faithful to Human Perception?”](https://arxiv.org/pdf/2309.13038.pdf)
>>>>>>> Stashed changes

[**Project Page**](https://sites.google.com/view/semsim)


---
The important experimental part can be found at ```benchmark/```.

The existing metrics can be found at ```metrics/```.
## Setup
You can use [anaconda](https://www.anaconda.com/distribution/) to install our setup by running
```
conda env create -f semsim.yml
conda activate semsim
```


## Getting Started
<<<<<<< Updated upstream
####  Step 1:  train classifier to be evaluated
=======
####  Step1:  train classifier to be evaluated
>>>>>>> Stashed changes
```
# using cifar-100 as example
python benchmark/step1_train_classifier.py --data=cifar100 --arch=ResNet20-4 --epochs=200 --aug_list='' --mode=crop
```


####  Step 2: attack classifier to get reconstructed images

Original images are needed in this step, you can 

* **use the provided [CIFAR-100 samples](https://drive.google.com/file/d/1TjRNUX5KTzEAXYVhCHROD5ZVE5uFNosE/view?usp=drive_link)**

   (1) download the [CIFAR-100 samples](https://drive.google.com/file/d/1TjRNUX5KTzEAXYVhCHROD5ZVE5uFNosE/view?usp=drive_link)
 
   (2) place the original images in the directory: `benchmark/images/Cifar_ori/`

```
# using cifar-100 as example,

python benchmark/step2_attack.py --data=cifar100 --arch=ResNet20-4 --epochs=200 --aug_list='' --mode=crop --optim='inversed'
```


* **use your own dataset:**

  (1) if you prefer to use your own dataset, place your dataset in your chosen directory.
  
  (2) set the '--rec_data_dir' parameter to the directory containing your dataset.


* **reconstructed images:**

   (1) you can also use our prepared reconstructed images for this step. 
   
   (2) download them from this [link](https://drive.google.com/file/d/12AXAPTTRyDfUJ3s807Oy-CxXk3E1Py9z/view?usp=sharing).

<<<<<<< Updated upstream
   (3) place the original images in the directory: `benchmark/images/cifar100/`

#### Step 3: use different metric to measure the privacy leakage
=======
#### Step3: use different metrics to measure the privacy leakage
>>>>>>> Stashed changes


* **Existing metric**
```
python metrics/pixel_level_metrics.py

<<<<<<< Updated upstream
# modify line 137-138 
=======
# modify Lines 137-138 
>>>>>>> Stashed changes
#    data_dir_raw= '' # dir of original images 
#    with open('metrics/folder_names_cifar.txt', 'r') as f: 
#    folder_names_cifar.txt saves dirs of reconstructed images 
```

* **SemSim**
```
<<<<<<< Updated upstream
# train SemSim. 
# Data path is set in the Line 205 of 'inversefed/data/data_processing.py'
python benchmark/Semsim_train_evaluation.py --data human_anno_id --arch ResNet18 --epochs 100 --mode crop --semsim True
=======
# train Semsim. 
# Data path is set in Line 205 of 'inversefed/data/data_processing.py'
python benchmark\Semsim_train_evaluation.py --data human_anno_id --arch ResNet18 --epochs 100 --mode crop --semsim True
>>>>>>> Stashed changes
```


```
# test SemSim
python benchmark/Semsim_train_evaluation.py --data human_anno_id --arch ResNet18 --epochs 100 --mode crop --semsim True --evaluate True

<<<<<<< Updated upstream
# '--target_data' is the target test set you want to evaluated. The default value is 'cifar100'.
=======
# '--targte_data' is the target test set you want to evaluate. The default value is 'cifar100'.
>>>>>>> Stashed changes
```

#### Step 4: analyze results

```
# calculate correlation between different metrics
python metrics/models_rank_corr.py 
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
We express gratitude to the great work [ATSPrivacy](https://github.com/gaow0007/ATSPrivacy), [Inverting Gradients](https://github.com/JonasGeiping/invertinggradients) and [DLG](https://github.com/mit-han-lab/dlg) as we benefit a lot from both their papers and codes.

# License
This repository is released under the MIT license. 
