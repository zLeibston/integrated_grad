import torch
from utils.plot_pic import plot_MNIST,plot_MNIST_denormalized,plot_CIFAR10,plot_CIFAR10_denormalized
from utils.load_data import load_MNIST,load_CIFAR10
from model_train.mnist_classifier import MnistClassifier
from model_train.cifar10_classfier import Cifar10_Classifier

def find_error_mnist(model, data_loader,num_images=16):
    model.eval()
    images_list = []
    labels_list = []
    predictions_list = []
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    images_list.append(images[i])
                    labels_list.append(labels[i].item())
                    predictions_list.append(predicted[i].item())
                if len(images_list) >= num_images:
                    break
            if len(images_list) >= num_images:
                break
    plot_MNIST(images_list, labels_list, predictions_list)
    plot_MNIST_denormalized(images_list, labels_list, predictions_list)
    

def find_error_cifar10(model, data_loader,num_images=16):
    model.eval()
    images_list = []
    labels_list = []
    predictions_list = []
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    images_list.append(images[i])
                    labels_list.append(labels[i].item())
                    predictions_list.append(predicted[i].item())
                if len(images_list) >= num_images:
                    break
            if len(images_list) >= num_images:
                break
    plot_CIFAR10(images_list, labels_list, predictions_list)
    plot_CIFAR10_denormalized(images_list, labels_list, predictions_list)

if __name__ == "__main__":
    #  MNIST的错误样本展示
    mnist_model = MnistClassifier()
    mnist_model.load_state_dict(torch.load("./models/best_mnist_classfier.pth"))
    mnist_test_loader = load_MNIST(isTrain=False)
    find_error_mnist(mnist_model, mnist_test_loader, num_images=16)

    # CIFAR-10的错误样本展示
    cifar10_model = Cifar10_Classifier()
    cifar10_model.load_state_dict(torch.load("./models/best_cifar10_classfier.pth"))
    cifar10_test_loader = load_CIFAR10(isTrain=False)
    find_error_cifar10(cifar10_model, cifar10_test_loader, num_images=16)

    