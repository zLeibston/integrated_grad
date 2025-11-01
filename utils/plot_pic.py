import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_MNIST(images, labels, predictions=None):
    """
    Plots a grid of MNIST images with their labels and optional predictions.
    
    Parameters:
    - images: A batch of images (tensor or numpy array) of shape (N, 1, 28, 28)
    - labels: True labels corresponding to the images
    - predictions: Optional predicted labels corresponding to the images
    """
    num_images = len(images)
    grid_size = int(np.ceil(np.sqrt(num_images)))
    
    plt.figure(figsize=(10, 10))#这个figure的数字的大小可以调节整体图像的大小
    
    for i in range(num_images):
        plt.subplot(grid_size, grid_size, i + 1)
        img = images[i].squeeze()  # Remove channel dimension
        plt.imshow(img, cmap='gray')
        title = f'True: {labels[i]}'
        if predictions is not None:
            title += f'\nPred: {predictions[i]}'
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def plot_CIFAR10(images, labels, predictions=None):
    """
    Plots a grid of CIFAR-10 images with their labels and optional predictions.
    
    Parameters:
    - images: A batch of images (tensor or numpy array) of shape (N, 3, 32, 32)
    - labels: True labels corresponding to the images
    - predictions: Optional predicted labels corresponding to the images
    """
    cifar10_classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    num_images = len(images)
    grid_size = int(np.ceil(np.sqrt(num_images)))
    
    plt.figure(figsize=(10, 10))#这个figure的数字的大小可以调节整体图像的大小
    
    for i in range(num_images):
        plt.subplot(grid_size, grid_size, i + 1)
        img = images[i].permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)
        plt.imshow(img)
        title = f'True: {cifar10_classes[labels[i]]}'
        if predictions is not None:
            title += f'\nPred: {cifar10_classes[predictions[i]]}'
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()




def plot_MNIST_denormalized(images, labels, predictions=None, 
                            mean=(0.1307,), 
                            std=(0.3081,)):
    
    num_images = len(images)
    grid_size = int(np.ceil(np.sqrt(num_images)))
        
    plt.figure(figsize=(10, 10))#这个figure的数字的大小可以调节整体图像的大小

    mean_tensor = torch.tensor(mean).to(images.device if isinstance(images, torch.Tensor) else 'cpu')
    std_tensor = torch.tensor(std).to(images.device if isinstance(images, torch.Tensor) else 'cpu')
        
    for i in range(num_images):
        plt.subplot(grid_size, grid_size, i + 1)
        img = images[i].squeeze() *mean_tensor + std_tensor  # 反标准化
        img = torch.clamp(img, 0, 1)  # 裁剪到有效范围
        img = img.cpu().numpy()
        plt.imshow(img, cmap='gray')
        title = f'True: {labels[i]}'
        if predictions is not None:
            title += f'\nPred: {predictions[i]}'
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()





def plot_CIFAR10_denormalized(images, labels, predictions=None, 
                              mean=(0.4914, 0.4822, 0.4465), 
                              std=(0.2023, 0.1994, 0.2010)):
    cifar10_classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    num_images = len(images)
    grid_size = int(np.ceil(np.sqrt(num_images)))
    
    plt.figure(figsize=(10, 10))
    
    # 将均值和标准差转换为PyTorch张量，并调整为 (3, 1, 1) 形状
    # 这样它们可以在通道维度上与 (3, H, W) 的图像进行广播
    # 确保它们与图像在同一设备上，以便进行计算
    # 在这个函数里，我们最终会把图像移到CPU，所以这里也移到CPU是安全的
    mean_tensor = torch.tensor(mean).view(3, 1, 1).to(images.device if isinstance(images, torch.Tensor) else 'cpu')
    std_tensor = torch.tensor(std).view(3, 1, 1).to(images.device if isinstance(images, torch.Tensor) else 'cpu')

    for i in range(num_images):
        plt.subplot(grid_size, grid_size, i + 1)
        
        # 确保图像在CPU上进行反标准化和绘图
        current_image = images[i].cpu() # 确保在CPU
        
        # 反标准化： img = img * std + mean
        img_denorm = current_image * std_tensor.cpu() + mean_tensor.cpu() # 确保 mean/std 也都在CPU
        
        # 将张量裁剪到有效像素值范围 [0, 1]
        img_denorm = torch.clamp(img_denorm, 0, 1)

        # 调整维度并转换为NumPy
        img_display = img_denorm.permute(1, 2, 0).numpy()
        
        plt.imshow(img_display)
        
        # 确保 labels 和 predictions 在CPU上转换为Python数字，以便打印
        true_label = labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i]
        title = f'True: {cifar10_classes[true_label]}'
        
        if predictions is not None:
            predicted_label = predictions[i].item() if isinstance(predictions[i], torch.Tensor) else predictions[i]
            title += f'\nPred: {cifar10_classes[predicted_label]}'
        plt.title(title)
        plt.axis('off') 

    plt.tight_layout() 
    plt.show()

def heat_map_mnist(data, title='mnist', cmap_name='RdBu_r'): # 默认使用 'RdBu_r'，_r 表示反转颜色
    """
    Plots a heatmap for the given 2D data, distinguishing positive and negative values.
    
    Parameters:
    - data: 2D array-like structure (tensor or numpy array)
    - title: Title of the heatmap
    - cmap_name: Name of the diverging colormap to use (e.g., 'RdBu', 'coolwarm')
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    

    if data.ndim == 3:
        print(f"Warning: Input data is 3D ({data.shape}). Taking the mean along channel dimension for 2D visualization.")
        data = np.mean(data, axis=0) 
    
    plt.figure(figsize=(8, 6))

  
    abs_max = np.max(np.abs(data))

    
    # 设置 vmin 和 vmax，以零为中心
    # RdBu_r 表示红色为负，蓝色为正，白色为零。如果需要蓝色为负，红色为正，用 'RdBu'。
    # coolwarm 也是一个很好的选择
    plt.imshow(data, cmap=cmap_name, vmin=-abs_max, vmax=abs_max, interpolation='nearest')
    
    plt.colorbar(label='Attribution Value') # 给颜色条添加标签
    plt.title(title)
    plt.show()


def heat_map_cifar10(data, title='cifar10', cmap_name='RdBu_r'): # 默认使用 'RdBu_r'，_r 表示反转颜色
  
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    if data.ndim != 3:
        raise ValueError(f"Input data must be 3D (C, H, W). Got shape: {data.shape}")
    
    plt.figure(figsize=(10, 5))
    
    for c in range(data.shape[0]):
        channel_data = data[c]

        plt.subplot(1, 3, c + 1)
    
        abs_max = np.max(np.abs(channel_data))

        plt.imshow(channel_data, cmap=cmap_name, vmin=-abs_max, vmax=abs_max, interpolation='nearest')
        
        plt.colorbar(label='Attribution Value') # 给颜色条添加标签
        plt.title(f'{title} - Channel {c}')
        plt.axis('off')
    plt.tight_layout() 
    plt.show()