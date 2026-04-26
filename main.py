import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image

from model import MLP
from criterion import CrossEntropyLoss
from optimizer import SGD
from data_loader import EuroSATDataLoader

def save_best_config(config, path):
    with open(path, 'w') as f:
        import json
        json.dump(config, f, indent=4)

def load_weights(model, weight_path):
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"未找到权重文件: {weight_path}")
    
    with open(weight_path, 'rb') as f:
        weights = pickle.load(f)
    
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'get_params_and_grads'):
            layer.W = weights[f'layer_{i}_W']
            layer.b = weights[f'layer_{i}_b']
    print(f"成功加载权重: {weight_path}")

def run_grid_search(dataloader, args):
    lrs = [0.01, 0.005]
    hidden_dims = [256, 512]
    reg_strengths = [1e-4, 1e-5]
    
    best_overall_acc = 0
    best_config = None

    from trainer import Trainer

    for lr in lrs:
        for h_dim in hidden_dims:
            for wd in reg_strengths:
                config_name = f"lr{lr}_h{h_dim}_wd{wd}"
                print(f"\n[Grid Search] 正在测试配置: {config_name}")
                
                model = MLP(input_dim=12288, hidden_dim=h_dim, num_classes=10, activation_type=args.activation)
                optimizer = SGD(model, lr=lr, weight_decay=wd, lr_decay=0.98)
                criterion = CrossEntropyLoss()
                
                save_path = os.path.join(args.exp_dir, config_name)
                trainer = Trainer(model, optimizer, criterion, dataloader, save_dir=save_path)
                val_acc = trainer.train(epochs=args.epochs, batch_size=args.batch_size)
                
                if val_acc > best_overall_acc:
                    best_overall_acc = val_acc
                    best_config = {"lr": lr, "hidden_dim": h_dim, "weight_decay": wd, "val_acc": val_acc}
                    save_best_config(best_config, os.path.join(args.exp_dir, "best_config.json"))

    print(f"\n网格搜索完成！最优验证集准确率: {best_overall_acc:.4f}")

def load_model(args):
    if not os.path.exists(args.weight_path):
        raise FileNotFoundError(f"未找到权重文件: {args.weight_path}")
        
    with open(args.weight_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    config = checkpoint.get('config', {})
    h_dim = config.get('hidden_dim', args.hidden_dim)
    act_type = config.get('activation_type', args.activation)
    input_dim = config.get('input_dim', 12288)
    num_classes = config.get('num_classes', 10)
    
    print(f"[Auto-Config] 加载 model ckpt: hidden_dim={h_dim}, activation={act_type}")

    model = MLP(input_dim=input_dim, 
                hidden_dim=h_dim, 
                num_classes=num_classes, 
                activation_type=act_type)
    
    weights = checkpoint.get('weights', checkpoint) # 兼容旧格式
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'get_params_and_grads'):
            layer.W = weights[f'layer_{i}_W']
            layer.b = weights[f'layer_{i}_b']

    return model

def test_and_visualize(dataloader, args):
    model = load_model(args)
    
    X_test, y_test = dataloader.get_full_data('test')
    logits = model.forward(X_test)
    preds = np.argmax(logits, axis=1)
    
    acc = np.sum(preds == y_test) / len(y_test)
    print(f"\n[Test Result] 独立测试集准确率: {acc:.4f}")
    
    error_analysis(dataloader, args, preds, X_test, y_test)

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=dataloader.classes, yticklabels=dataloader.classes, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix on EuroSAT Test Set')
    plt.savefig(os.path.join(args.exp_dir, "confusion_matrix.png"))
    plt.close()
    print(f"混淆矩阵已保存至: {args.exp_dir}/confusion_matrix.png")
    
    visualize_first_layer_weights(model, dataloader.mean, dataloader.std, args.exp_dir)

def error_analysis(dataloader, args, preds, X_test, y_test):
    results_file = os.path.join(args.exp_dir, "test_predictions.csv")
    with open(results_file, 'w') as f:
        f.write("Index,True_Label,Predicted_Label,Is_Correct\n")
        for i in range(len(y_test)):
            true_name = dataloader.classes[y_test[i]]
            pred_name = dataloader.classes[preds[i]]
            is_correct = "True" if y_test[i] == preds[i] else "False"
            f.write(f"{i},{true_name},{pred_name},{is_correct}\n")
    print(f"测试集预测清单已保存至: {results_file}")

    error_indices = np.where(preds != y_test)[0]
    error_dir = os.path.join(args.exp_dir, "error_analysis")
    os.makedirs(error_dir, exist_ok=True)
    
    print(f"发现 {len(error_indices)} 个分类错误，导出前 10 个错例图像")
    for idx in error_indices[:10]:
        true_name = dataloader.classes[y_test[idx]]
        pred_name = dataloader.classes[preds[idx]]
        
        img_flat = X_test[idx] * dataloader.std + dataloader.mean
        img_array = img_flat.reshape(64, 64, 3)
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        img = Image.fromarray(img_array)
        img_name = f"idx{idx}_True_{true_name}_Pred_{pred_name}.jpg"
        img.save(os.path.join(error_dir, img_name))
    print(f"错例图像已保存至: {error_dir}")

def visualize_first_layer_weights(model, mean, std, exp_dir):
    W1 = model.layers[0].W
    num_neurons = 16
    plt.figure(figsize=(12, 12))
    
    for i in range(num_neurons):
        weight_vector = W1[:, i]
        
        # 逆向操作：反归一化, 有助于观察真实色彩倾向
        weight_img = weight_vector * std + mean 
        
        # Reshape 回图像尺寸 (64, 64, 3)
        weight_img = weight_vector.reshape(64, 64, 3)
        
        # 归一化到 [0, 1]
        low, high = np.min(weight_img), np.max(weight_img)
        weight_img = (weight_img - low) / (high - low)
        
        plt.subplot(4, 4, i + 1)
        plt.imshow(weight_img)
        plt.axis('off')
        plt.title(f"Neuron {i}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "weight_visualization.png"))
    plt.show()
    print(f"权重可视化已保存至: {exp_dir}/weight_visualization.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EuroSAT MLP Assignment")
    parser.add_argument('--mode', type=str, choices=['search', 'test'], required=True, help="运行模式: search (网格搜索) 或 test (测试及可视化)")
    parser.add_argument('--data_dir', type=str, default='./EuroSAT_RGB', help="数据集路径")
    parser.add_argument('--exp_dir', type=str, default='./runs', help="实验结果保存路径")
    parser.add_argument('--weight_path', type=str, default='./runs/best_model.pkl', help="测试时加载的权重路径")
    
    # 模型及训练超参数
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    
    args = parser.parse_args()
    
    dataloader = EuroSATDataLoader(args.data_dir)
    
    if args.mode == 'search':
        run_grid_search(dataloader, args)
    elif args.mode == 'test':
        test_and_visualize(dataloader, args)