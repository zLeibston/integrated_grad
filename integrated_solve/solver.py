
import torch
from utils.load_data import get_CIFAR10_dataset, get_MNIST_dataset
import torch.nn as nn
from model_train.mnist_classifier import MnistClassifier
from model_train.cifar10_classfier import Cifar10_Classifier
from model_train.mnist_classifier_plus import MnistClassifierPlus
from utils.plot_pic import plot_MNIST_denormalized, plot_CIFAR10_denormalized,heat_map_cifar10,heat_map_mnist



class SampleProvider:
    def __init__(self,datasample):
        self.datasample = datasample
        self.len = self.datasample.__len__()
        self.shape = self.datasample[0][0].shape  #都是（C,H,W）
        C, H, W = self.shape

        if(C==1):
            mean = torch.tensor([0.1307]).view(C, 1, 1)
            std = torch.tensor([0.3081]).view(C, 1, 1)
        else:
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(C, 1, 1)
            std = torch.tensor([0.247, 0.243, 0.261]).view(C, 1, 1)

        self.baselines = {}
       
        self.baselines['black'] = (torch.zeros((C, H, W))-mean)/std
        self.baselines['white'] = (torch.ones((C, H, W))-mean)/std

        guassian_baseline = torch.randn((C, H, W)) 
        guassian_baseline = torch.clamp(guassian_baseline,0,1)
        self.baselines['guassian'] = (guassian_baseline-mean)/std

        average_baseline = torch.zeros((C, H, W))
        for _ in range(self.len):
            random_img = self.datasample[0][torch.randint(0,self.len,(1,)).item()]
            average_baseline += random_img
        average_baseline /= self.len
        self.baselines['average'] = average_baseline
  
    def baseline_selector(self,baseline_type='black'):
        return self.baselines.get(baseline_type)
        
    def random_sample(self,num_samples=1)-> tuple[list[torch.Tensor],list[int]]:
        samples = []
        labels = [] 
        for _ in range(num_samples):
            idx = torch.randint(0,self.len,(1,)).item()
            samples.append(self.datasample[0][idx])
            labels.append(self.datasample[1][idx])
        return samples,labels
       



class Solver:
    def __init__(self, model, sampleprivider: SampleProvider, num_step: int = 100):
        self.sampleprivider = sampleprivider
        self.model = model
        self.shape = sampleprivider.shape
        self.time_coeff = [i *1.0/ num_step for i in range(num_step + 1)]

    

    def fusion_of_imgs(self,baseline_type,time_step,img):
        baseline = self.sampleprivider.baseline_selector(baseline_type)
        fused_img = baseline + self.time_coeff[time_step] * (img - baseline)
        return fused_img


    def grad_of_imgs(self,img:torch.Tensor,target_label:int):
        img = img.clone().detach().unsqueeze(0).requires_grad_(True)

        self.model.eval()

        output = self.model(img)
        
        value = output[0,target_label]

        self.model.zero_grad()
        if img.grad is not None:
            img.grad.zero_()
        value.backward()
    
        grad = None
        if img.grad is not None:
            grad = img.grad.data.squeeze(0)  #移除batch维度
        return grad
    
    def compute_integrated_gradients(self,baseline_type):
        list_img ,list_label = self.sampleprivider.random_sample(num_samples=1)
        img ,label= list_img[0], list_label[0]

        total_grad = torch.zeros_like(img)

        for t in range(len(self.time_coeff)):
            fused_img = self.fusion_of_imgs(baseline_type, t, img)
            grad = self.grad_of_imgs(fused_img, label)
            total_grad += grad

        integrated_gradients = (img - self.sampleprivider.baseline_selector(baseline_type)) * total_grad / len(self.time_coeff)

        return integrated_gradients,img,label
        


def select_samples(dataset, num_samples=16):
    images_list = []
    labels_list = []

    random_indices = torch.randperm(len(dataset))[:num_samples]

    for idx in random_indices:
        image, label = dataset[idx]
        images_list.append(image)
        labels_list.append(label)

    return (images_list, labels_list)




if __name__ == "__main__":
    num_samples = 1000
    num_step = 1000

    mnist_sample = select_samples(get_MNIST_dataset(isTrain=False),num_samples)
    cifar10_sample = select_samples(get_CIFAR10_dataset(isTrain=False),num_samples)

    mnist_model = MnistClassifier()
    mnist_model_plus = MnistClassifierPlus()
    cifar10_model = Cifar10_Classifier()

    mnist_model.load_state_dict(torch.load("models/best_mnist_classfier.pth"))
    mnist_model_plus.load_state_dict(torch.load("models/best_mnist_classfier_plus.pth"))
    cifar10_model.load_state_dict(torch.load("models/best_cifar10_classfier.pth"))

    mnist_provider = SampleProvider(mnist_sample)
    cifar10_provider = SampleProvider(cifar10_sample)

    mnist_solver_plus = Solver(mnist_model_plus, mnist_provider,num_step)
    cifar10_solver = Solver(cifar10_model, cifar10_provider,num_step)

    mnist_ig_b ,img1 ,label1 = mnist_solver_plus.compute_integrated_gradients(baseline_type='black')
    mnist_ig_g ,img2 ,label2 = mnist_solver_plus.compute_integrated_gradients(baseline_type='guassian')
    mnist_ig_a ,img3 ,label3 = mnist_solver_plus.compute_integrated_gradients(baseline_type='average')
    mnist_ig_w ,img4 ,label4 = mnist_solver_plus.compute_integrated_gradients(baseline_type='white')



    cifar10_ig_b ,img5 , label5 = cifar10_solver.compute_integrated_gradients(baseline_type='black')
    cifar10_ig_g ,img6 , label6= cifar10_solver.compute_integrated_gradients(baseline_type='guassian')
    cifar10_ig_a ,img7 , label7= cifar10_solver.compute_integrated_gradients(baseline_type='average')
    cifar10_ig_w ,img8 , label8= cifar10_solver.compute_integrated_gradients(baseline_type='white')

    plot_MNIST_denormalized(torch.stack([mnist_ig_b,img1]),[label1,label1])
    heat_map_mnist(mnist_ig_b)
    plot_MNIST_denormalized(torch.stack([mnist_ig_g,img2]),[label2,label2])
    heat_map_mnist(mnist_ig_g)
    plot_MNIST_denormalized(torch.stack([mnist_ig_a,img3]),[label3,label3])
    heat_map_mnist(mnist_ig_a)
    plot_MNIST_denormalized(torch.stack([mnist_ig_w,img4]),[label4,label4])
    heat_map_mnist(mnist_ig_w)




    plot_CIFAR10_denormalized(torch.stack([cifar10_ig_b,img5]),[label5,label5])
    heat_map_cifar10(cifar10_ig_b)
    plot_CIFAR10_denormalized(torch.stack([cifar10_ig_g,img6]),[label6,label6])
    heat_map_cifar10(cifar10_ig_g)
    plot_CIFAR10_denormalized(torch.stack([cifar10_ig_a,img7]),[label7,label7])
    heat_map_cifar10(cifar10_ig_a)
    plot_CIFAR10_denormalized(torch.stack([cifar10_ig_w,img8]),[label8,label8])
    heat_map_cifar10(cifar10_ig_w)



 