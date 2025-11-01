
import torch
from utils.load_data import get_CIFAR10_dataset, get_MNIST_dataset
import torch.nn as nn
from model_train.mnist_classifier import MnistClassifier
from model_train.cifar10_classfier import Cifar10_Classifier
from utils.plot_pic import plot_MNIST, plot_CIFAR10_denormalized    

class Solver:
    def __init__(self,dataset,model):
        self.dataset = dataset
        self.model = model
        self.len = self.dataset.__len__()
        self.shape = self.dataset[0][0].shape  #都是（C,H,W）
        



def select_samples(dataset, num_samples=16):
    images_list = []
    labels_list = []

    random_indices = torch.randperm(len(dataset))[:num_samples]

    for idx in random_indices:
        image, label = dataset[idx]
        images_list.append(image)
        labels_list.append(label)

    return images_list, labels_list







if __name__ == "__main__":
    num_samples = 16

    mnist_sample = select_samples(get_MNIST_dataset(isTrain=False),num_samples)
    cifar10_sample = select_samples(get_CIFAR10_dataset(isTrain=False),num_samples)

    mnist_model = MnistClassifier()
    cifar10_model = Cifar10_Classifier()

    mnist_solver = Solver(mnist_sample, mnist_model)
    cifar10_solver = Solver(cifar10_sample, cifar10_model)

    print(mnist_solver.shape)
    print(cifar10_solver.shape)