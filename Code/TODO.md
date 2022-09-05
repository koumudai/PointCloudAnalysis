# TODO

## 缩写

- `cls`:
- `part_seg`:
- `sem_seg`:

## 数据集

- `ModelNet`:
- `ScanObjectNN`:
- `ShapeNet_55_34`:
- `ShapeNet`:
- `S3DIS`:

## 模型

- `PointNet`:
- `PointNet++`(`PointNet2`):
- `DGCNN`:
- `PointTransformer`:
- `AdaptConv`:

## 框架

### 主框架

- `configs`: 存放数据集或模型的配置信息, 配置信息格式为 `yaml`, 格式参考.
- `data`: 存放数据集及类别信息.
- `datasets`: 存放 `Dataset` 类.
- `models`: 存放 `Model` 类.
- `losses`: 存放 `Loss` 类.
- `optimizers`: 存放 `Optimizer` 类.
- `schedulers`: 存放 `Scheduler` 类
- `tools`: 存放 `runner` 和 `builder` 函数.
- `utils`: 存放一些重要的函数.
- `main`: 存放 `main` 函数.

### 全部框架

- `configs`: 存放数据集或模型的配置信息, 配置信息格式为 `yaml`.
  - `datasets`: 存放数据集的配置信息, 其下文件夹命名规范为 `datasets_{dataset_base_name}`.
    - `datasets_ModelNet`: 存放与数据集 `ModelNet` 相关的配置信息.
    - `datasets_ScanObjectNN`: 存放与数据集 `ScanObjectNN` 相关的配置信息.
    - `datasets_ShapeNet_55_34`: 存放与数据集 `ShapeNet_55_34` 相关的配置信息.
  - `models_cls`: 存放与 `cls` 任务相关的模型配置信息, 其下文件夹命名规范为 `models_{dataset_base_name}`.
    - `models_ModelNet`: 存放与 `ModelNet` 数据集相关的模型配置信息, 其下文件夹命名规范为 `{model_name}_{dataset_name}_{n_point}pts.yaml`.
      - `DGCNN_ModelNet40_1024pts.yaml`:
    - `models_ScanObjectNN`:
    - `models_ShapeNet_55_34`:
  - `models_part_seg`: 存放与 `part_seg` 任务相关的模型配置信息, 其下文件夹命名规范为 `models_{dataset_base_name}`.
    - `models_ShapeNet`:
  - `models_sem_seg`: 存放与 `sem_seg` 任务相关的模型配置信息, 其下文件夹命名规范为 `models_{dataset_base_name}`.
    - `models_S3DIS`:
- `data`: 存放数据集
  - `ModelNet`: 存放 `ModelNet` 数据集的信息.
    - `ModelNet10`:
    - `ModelNet40`:
  - `ScanObjectNN`: 存放 `ScanObjectNN` 数据集的信息.
    - `ScanObjectNN_OBJ_ONLY`:
    - `ScanObjectNN_OBJ_ONLY_PB_T25`:
    - `ScanObjectNN_OBJ_ONLY_PB_T25_R`:
    - `ScanObjectNN_OBJ_ONLY_PB_T50_R`:
    - `ScanObjectNN_OBJ_ONLY_PB_T50_RS`:
    - `ScanObjectNN_OBJ_BG`:
    - `ScanObjectNN_OBJ_BG_PB_T25`:
    - `ScanObjectNN_OBJ_BG_PB_T25_R`:
    - `ScanObjectNN_OBJ_BG_PB_T50_R`:
    - `ScanObjectNN_OBJ_BG_PB_T50_RS`:
  - `ShapeNet_55_34`: 存放 `ShapeNet_55_34` 数据集的信息.
    - `ShapeNet55`:
  - `ShapeNet`: 存放 `ShapeNet` 数据集的信息.
    - `ShapeNet`:
  - `S3DIS`: 存放 `S3DIS` 数据集的信息.
    - `S3DIS`:
- `datasets`:
  - `__init__.py`:
  - `build.py`:
  - `ModelNetDataset.py`:
  - `ScanObjectNNDataset.py`:
  - `ShapeNet55Dataset.py`:
  - `ShapeNetDataset.py`:
- `models`:
  - `__init__.py`:
  - `build.py`:
  - `pointcloud_utils.py`:
  - `PointNet`:
    - `pointnet_utils.py`:
    - `pointnet_cls.py`:
    - `pointnet_part_seg.py`:
    - `pointnet_sem_seg.py`:
  - `PointNet2`:
    - `pointnet2_utils.py`:
    - `pointnet2_cls_ssg.py`:
    - `pointnet2_cls_msg.py`:
    - `pointnet2_part_seg_ssg.py`:
    - `pointnet2_part_seg_msg.py`:
    - `pointnet2_sem_seg_ssg.py`:
    - `pointnet2_sem_seg_msg.py`:
  - `DGCNN`:
    - `dgcnn_utils.py`:
    - `dgcnn_cls.py`:
    - `dgcnn_part_seg.py`:
    - `dgcnn_sem_seg.py`:
  - `PointTransformer`:
    - `pointtransformer_utils.py`:
    - `pointtransformer_cls.py`:
    - `pointtransformer_part_seg.py`:
    - `pointtransformer_sem_seg.py`:
  - `AdaptConv`:
    - `adaptconv_utils.py`:
    - `adaptconv_cls.py`:
    - `adaptconv_part_seg.py`:
    - `adaptconv_sem_seg.py`:
- `losses`:
  - `__init__.py`:
  - `build.py`:
  - `PointNet`:
    - `PointNetLoss.py`
  - `PointNet2`:
    - `PointNet2Loss.py`
  - `DGCNN`:
  - `PointTransformer`:
  - `AdaptConv`:
- `optimizers`:
- `scheduler`:
- `tools`:
  - `__init__.py`:
  - `build.py`:
- `utils`:
- `main`:
  - `main_cls.py`:
  - `main_part_seg.py`:
  - `main_sem_seg.py`:
- `DATASET.md`
- `README.md`
- `requirements.txt`

<!-- ## Yaml Formatter

```bash
python ./yaml_formatter.py
``` -->

## TODO List

1. `registry.py` 需要重写, 以满足 `./tools/builder.py`, `configs/models` 下的每个 `yaml` 配置文件中的 `dataset` 属性的结构需要全部修改.
2. 需要修改 `dataset` 中的返回值, 将其改成字典
3. 

模型的输入为: feature, coord
模型的输出为: logits, rtkwargs


## Requirement

```
conda create -n pca python=3.7
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

## References

[1] [PointNet](https://arxiv.org/abs/1612.00593)



```yaml
args:
  cfg_file: str
  use_gpu: True
  gpu: 4, 5, 6, 7
  num_workers: 8
  seed: 0

```