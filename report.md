# HW1：从零开始构建三层神经网络分类器，实现 地表覆盖图像分类

### 项目链接

[Arksuzuran/CS60003-hw1-EuroSAT](https://github.com/Arksuzuran/CS60003-hw1-EuroSAT/tree/main)

**代码**

```
GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:Arksuzuran/CS60003-hw1-EuroSAT.git
```

**最佳ckpt**

模型权重使用 Git LFS 管理

```
git lfs install
git lfs pull --include="runs/lr0.01_h512_wd0.0001/best_model.pkl"
```

### 1. 数据集处理

本次实验使用 EuroSAT 遥感图像数据集，该数据集包含 10 个不同的地表覆盖类别（如森林、河流、高速公路、住宅区等）。为了保证实验的严谨性与模型泛化能力的客观评估，整个数据预处理流程如下：

1.  **数据展平**：原始图像分辨率为 $64 \times 64$ 像素，包含 RGB 三个通道。在加载阶段，每张图像被展平为一维向量，输入特征维度为 $64 \times 64 \times 3 = 12288$。
2.  **分层采样与数据集划分**：为保证各类别在训练和验证过程中的分布平衡，对每个类别内部进行了打乱，并按照 70% : 15% : 15% 的比例划分为训练集、验证集 和独立测试集。划分结果被持久化保存到了`dataset_split.json`，以确保实验的可复现性。
3.  **防止数据泄露的归一化 (Z-score Normalization)**：仅计算**训练集**图像特征的均值（Mean）和标准差（Std）。随后，使用训练集的统计量对训练集、验证集和测试集进行全局归一化，有效避免了测试阶段的数据泄露，并加速了梯度下降的收敛。

### 2. 模型结构与优化逻辑

本实验全程不依赖 PyTorch、TensorFlow 等支持自动微分的现代深度学习框架，完全基于 NumPy 手工实现了矩阵运算、前向传播与反向传播计算图。

#### 2.1 MLP结构 

实验构建了一个标准的三层多层感知机（包含两个隐藏层和一个输出层）。模型支持自定义隐藏层维度（Hidden Dimension），并实现了 ReLU、Sigmoid 和 Tanh 等多种激活函数的动态切换。
模型的正向计算链路为：`Input -> Linear -> Activation -> Linear -> Activation -> Linear -> Output`。在反向传播中，严格按照链式法则，通过矩阵乘法将上游梯度逐层回传。

#### 2.2 优化器与损失函数
* **损失函数 (Cross-Entropy Loss)**：针对多分类任务，实现了数值稳定的 Softmax 激活结合交叉熵损失 。通过结合求导优化，将最后一层的梯度回传公式极简为 $\frac{1}{N}(P - Y)$，大幅提升了反向传播的计算效率与数值稳定性。
* **优化算法 (SGD with Weight Decay)**：实现了支持 L2 正则化（Weight Decay）的随机梯度下降优化器。在参数更新阶段，额外对权重矩阵引入惩罚项，以抑制过拟合。
* **学习率衰减 (Learning Rate Decay)**：引入了指数级学习率衰减策略，在每个 Epoch 结束后按比例衰减学习率，帮助模型在训练后期更好地收敛到局部最优解。代码并实现了基于验证集准确率自动保存最优权重的机制。

### 3. 项目结构设计

采用了面向对象设计，主要模块划分如下：
* `model.py`：定义了计算图节点基类 `Layer`，并派生出带有可学习参数的 `Linear` 层、无参数的激活函数层（`ReLU`, `Sigmoid`, `Tanh`）。`MLP` 类负责容器化组装。
* `optimizer.py`：包含 `SGD` 优化器，负责接收网络的可学习参数引用，并执行带有 L2 正则化的数学步进更新与梯度清零。
* `data_loader.py`：实现 `EuroSATDataLoader`，负责数据集图片的加载、划分、归一化以及生成批次 (Batch) 数据。
* `main.py` & `train.py`：串联模型、数据和优化器，实现了网格搜索、日志记录、模型保存及测试集评估的可视化等顶层逻辑。

---

### 4. 实验结果与超参数查找

#### 4.1 超参数网格搜索
实验利用网格搜索策略评估了不同超参数组合对模型性能的影响。搜索空间包括：学习率 $\eta \in$ `{0.01, 0.005}`，隐藏层大小 $H \in$ `[256, 512]`，L2 正则化强度 $\lambda \in$ `[1e-4, 1e-5]`。

**超参数搜索结果记录**

![combined_learning_curves](https://arksuzuran.oss-cn-beijing.aliyuncs.com/img/md_img/combined_learning_curves.png)

最终根据验证集准确率，选定的最优模型配置为：lr=0.01, h=512, wd=1e-4。

#### 4.2 学习曲线可视化
基于上述最优配置进行完整训练，模型在训练集和验证集上的 Loss 曲线与 Accuracy 曲线如下：

![learning_curves](https://arksuzuran.oss-cn-beijing.aliyuncs.com/img/md_img/learning_curves.png)
*最优配置下的训练集与验证集 Loss (左) 与 Accuracy (右) 曲线*

#### 4.3 独立测试集评估
导入验证集上保存的最优权重对独立测试集进行评估，最终分类准确率达到：**`[填写测试集 Accuracy，例如 85.4%]`**。各分类的混淆矩阵如下所示：

![confusion_matrix](https://arksuzuran.oss-cn-beijing.aliyuncs.com/img/md_img/confusion_matrix.png)
*测试集分类混淆矩阵*

---

### 5. 权重可视化与空间模式观察

为探究模型是否学习到了具备物理意义的底层特征，实验将第一层隐藏层的权重矩阵 $W_1$ 提取出部分神经元，恢复至原始图像尺寸 $(64 \times 64 \times 3)$ 并进行了反归一化可视化。

![weight_visualization](https://arksuzuran.oss-cn-beijing.aliyuncs.com/img/md_img/weight_visualization.png)
*第一层隐藏层部分神经元的权重可视化图*

**观察与讨论：**
基于可视化的权重图像，可以观察到以下现象：

1.  **色彩倾向提取**：神经元Neuron 0, Neuron  4呈现出强烈的绿色或棕褐色倾向。这表明模型在浅层成功提取了对应“森林 (Forest)”或“农田 (AnnualCrop)”的全局色彩基元。
2.  **空间纹理捕捉**：在 Neuron 2 的权重中，可以隐约观察到贯穿对角线的条带状空间模式，这可能是模型为识别“高速公路 (Highway)”或“河流 (River)”类别的线性边缘特征而形成的特定感受野响应。
3.  **局限性分析**：与卷积神经网络 (CNN) 具有局部平移不变性的卷积核不同，MLP 的权重展示出较强的全局性，这也解释了为何其对具有特定几何形状（如建筑物局部）的判别能力有限。

---

### 6. 错例分析

通过比对测试集的真实标签与预测结果，挑选出以下几个典型的分类错误样本进行分析：

**错误原因剖析：**

1. **形态与色彩的高度相似性**：

   ![idx3_True_AnnualCrop_Pred_River](https://arksuzuran.oss-cn-beijing.aliyuncs.com/img/md_img/idx3_True_AnnualCrop_Pred_River.jpg)

   该错例真实为农田 (AnnualCrop)，被误判为 River。从视觉上看，这两者在低分辨率下均呈现出灰黑色的细长带状连通结构。由于 MLP 将图像展平，丢失了图像的二维空间连续性，模型只能依赖全局色彩统计（如灰色像素比例）进行粗略判定，难以区分水波纹理与柏油路面，从而导致混淆。

2. **语义重叠与类间方差小**：

   ![idx5_True_AnnualCrop_Pred_PermanentCrop](https://arksuzuran.oss-cn-beijing.aliyuncs.com/img/md_img/idx5_True_AnnualCrop_Pred_PermanentCrop.jpg)

   该错例真实为农田 (AnnualCrop)，被误判为 PermanentCrop。地物特征上，两者均存在大面积的绿色草地覆盖。由于不存在明显的结构边界，仅依靠 MLP 提取的浅层 RGB 像素级特征，模型极难在这两个具有包含/重叠关系的语义类别中划清决策边界。

### 7. 结论

我们从零开始，使用 NumPy 构建了包含自动微分引擎的三层 MLP 分类器，成功完成了 EuroSAT 数据集的十分类任务。通过严谨的数据划分、网格搜索寻优以及可视化分析，验证了 MLP 模型在遥感图像宏观分类上的有效性，并通过权重可视化和错例分析探讨了全连接层在处理空间视觉任务时的特征提取机制与局限性。
