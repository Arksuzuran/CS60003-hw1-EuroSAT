import os
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

class Trainer:
    def __init__(self, model, optimizer, criterion, dataloader, save_dir="exp_results"):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 统计数据
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }

    def train(self, epochs, batch_size):
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in self.dataloader.get_batch('train', batch_size):
                self.optimizer.zero_grad()
                logits = self.model.forward(batch_X)
                loss = self.criterion.forward(logits, batch_y)
                
                grad_logits = self.criterion.backward()
                self.model.backward(grad_logits)
                
                self.optimizer.step()
                
                # 累加统计
                epoch_loss += loss * batch_X.shape[0]
                preds = np.argmax(logits, axis=1)
                correct += np.sum(preds == batch_y)
                total += batch_X.shape[0]
            
            # 计算训练集平均指标
            train_loss = epoch_loss / total
            train_acc = correct / total
            
            # --- 验证阶段 ---
            val_loss, val_acc = self.evaluate('val')
            self.optimizer.step_lr() # 学习率衰减
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            # 保存最优权重
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint("best_model.pkl")

        # 训练结束，绘制图表
        self.plot_curves()
        return best_val_acc

    def evaluate(self, split):
        X, y = self.dataloader.get_full_data(split)
        logits = self.model.forward(X)
        loss = self.criterion.forward(logits, y)
        preds = np.argmax(logits, axis=1)
        acc = np.sum(preds == y) / len(y)
        return loss, acc

    def save_checkpoint(self, filename):
        path = os.path.join(self.save_dir, filename)
        
        # 1. 提取权重
        weights = {}
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'get_params_and_grads'):
                weights[f'layer_{i}_W'] = layer.W
                weights[f'layer_{i}_b'] = layer.b
                
        # 2. 封装模型元数据 (Metadata)
        checkpoint = {
            'weights': weights,
            'config': {
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_classes': self.model.num_classes,
                'activation_type': self.model.activation_type
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"模型及配置已保存至: {path}")

    def plot_curves(self):
        epochs = range(1, len(self.history['train_loss']) + 1)
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], label='Train Loss')
        plt.plot(epochs, self.history['val_loss'], label='Val Loss')
        plt.title('Loss Curve')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['train_acc'], label='Train Acc')
        plt.plot(epochs, self.history['val_acc'], label='Val Acc')
        plt.title('Accuracy Curve')
        plt.legend()
        
        plt.savefig(os.path.join(self.save_dir, "learning_curves.png"))
        plt.close()