# SCL Training & validation



## Prepare

### Step 1. prepare for data



You need to process the WSI into the following format. The processing method can be found in  https://github.com/mahmoodlab/CLAM

The **CTransPath** feature extractor and the pretrained model can be download in  https://github.com/Xiyue-Wang/TransPath



```
DATA_DIR
├─patch_coord
│      slide_id_1.h5
│      slide_id_2.h5
│      ...
└─patch_feature
        slide_id_1.pt
        slide_id_2.pt
        ...
```

The h5 file in the `patch_coord` folder contains the coordinates of each patch of the WSI, which can be read as

```python
coords = h5py.File(coords_path, 'r')['coords'][:]
# coords is a array like:
# [[x1, y1], [x2, y2], ...]
```

The pt file in the `patch_feature`folder contains the features of each patch of the WSI, which can be read as

```python
features = torch.load(features_path, map_location=torch.device('cpu'))
# features is a tensor with dimension N*F, and if features are extracted using CTransPath, F is 768
```

### Step 2. preparing the data set split

You need to divide the dataset into a training set validation set and a test set, and store them in the following format

```
SPLIT_DIR
    test_set.csv
    train_set.csv
    val_set.csv
```

And, the format of the csv file is as follows

| slide_id   | label |
| ---------- | ----- |
| slide_id_1 | 0     |
| slide_id_2 | 1     |
| ...        | ...   |

## Train model

### Step 1. create a config file

We have prepared two config file templates (see ./configs/) for SCL-WC, like

```yaml
General:
    seed: 7
    work_dir: WORK_DIR
    fold_num: 4

Data:
    split_dir: SPLIT_DIR
    data_dir_1: DATA_DIR_1 
    features_size: 768
    n_classes: 2

Model:
    network: 'SCL'
```

In the config, the correspondence between the `Model.network`, `Train.training_method` and `Train.val_method` is as follows

| `Model.network` | `Train.training_method` | `Train.val_method` |
|-----------------|-------------------------|--------------------|
| SCL             | SCL                     | SCL                |

### Step 2. train model

Run the following command

```shell
python train.py --config_path [config path] --begin [begin index] --end [end index]
```

`--begin` and `--end` used to control repetitive experiments


## License

RetCCL is released under the GPLv3 License and is available for non-commercial academic purposes.

### Citation
Please use below to cite this paper if you find our work useful in your research.
```
@article{wang2022scl,
  title={Scl-wc: Cross-slide contrastive learning for weakly-supervised whole-slide image classification},
  author={Wang, Xiyue and Xiang, Jinxi and Zhang, Jun and Yang, Sen and Yang, Zhongyi and Wang, Ming-Hui and Zhang, Jing and Yang, Wei and Huang, Junzhou and Han, Xiao},
  journal={Advances in neural information processing systems},
  volume={35},
  pages={18009--18021},
  year={2022}
}
``` 



