# CS60003-hw1-EuroSAT

### 环境依赖

```
conda create env -f environment.yaml
```

### 项目结构

```
/CS60003-hw1-EuroSAT
│── main.py             	# 入口, 网格搜索或运行最优模型
│── model.py            	# Layer, Linear, Activation, MLP, CrossEntropyLoss
│── trainer.py				# 训练循环
│── optimizer.py        	# SGD 实现
│── criterion.py      		# 交叉熵损失
│── data_loader.py      	# EuroSAT 加载器与预处理
│── dataset_split.json  	# 数据集划分持久化文件
│── dataset_split.jsonenvironment.yaml	# conda 环境依赖
│── runs/               	# 自动生成的实验结果目录
│   └── lr0.01_h512_wd0.0001/
│       ├── best_model.pkl  # 最优权重
│       └── learning_curves.png 		# Loss/Acc 曲线图
└── README.md           	# 按作业要求注明环境依赖与运行方式
```


### 复现结果
```bash
conda activate eurosat-mlp
```

#### 执行网格搜索以寻找最优超参
输出目录为`runs`
```
python main.py --mode search
```

#### 模型测试与可视化

模型权重使用 Git LFS 管理, 使用git lfs拉取`ckpt`文件：

```
git lfs install
git lfs pull --include="runs/lr0.01_h512_wd0.0001/best_model.pkl"
```

输出目录为`runs/lr0.01_h512_wd0.0001`
```
python main.py --mode test --weight_path ./runs/lr0.01_h512_wd0.0001/best_model.pkl
```