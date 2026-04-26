import os
import glob
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def combine_learning_curves(exp_dir='./runs', output_name='combined_learning_curves.png'):
    search_pattern = os.path.join(exp_dir, '*', 'learning_curves.png')
    image_paths = glob.glob(search_pattern)
    
    if not image_paths:
        print(f"在 {exp_dir} 目录下没有找到 learning_curves.png")
        return
    
    image_paths.sort()
    
    n_images = len(image_paths)
    print(f"共找到 {n_images} 张曲线图")
    
    cols = 2
    rows = math.ceil(n_images / cols)
    
    fig_width = cols * 10
    fig_height = rows * 4.5
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    for i, img_path in enumerate(image_paths):
        config_name = os.path.basename(os.path.dirname(img_path))
        
        img = mpimg.imread(img_path)
        
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"Config: {config_name}", fontsize=16, fontweight='bold', pad=10)
        
    for j in range(n_images, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    output_path = os.path.join(exp_dir, output_name)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    combine_learning_curves()