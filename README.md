# CS60003-hw1-EuroSAT


### 环境依赖
使用numpy实现矩阵运算, pillow读入、预处理数据集, matplotlib, seaborn绘图

```
conda create env -f environment.yaml
```

### 项目结构


### 复现结果
```
conda activate eurosat-mlp
```

#### 执行网格搜索以寻找最优超参
输出目录为`runs`
```
bash scripts/train.sh
```

#### 模型测试与可视化
输出目录为`runs/lr0.01_h512_wd0.0001`
```
bash scripts/eval.sh
```