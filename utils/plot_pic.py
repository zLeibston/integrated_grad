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