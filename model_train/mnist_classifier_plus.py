import torch
from utils.load_data import load_MNIST
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

class MnistClassifierPlus(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Conv2d(1, 30, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(30)
        self.pool = nn.MaxPool2d(2, 2)

        self.cnn2 = nn.Conv2d(30, 40, kernel_size=3, padding=2)
        self.bn2 = nn.BatchNorm2d(40)
        self.pool = nn.MaxPool2d(2, 2)

        self.cnn3 = nn.Conv2d(40, 60, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(60)
        self.pool = nn.MaxPool2d(2, 2)

        self.cnn4 = nn.Conv2d(60, 80, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(80)
        self.pool = nn.MaxPool2d(2, 2)


        self.fc1 = nn.Linear(320, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, 50)
        self.fc4 = nn.Linear(50, 10)
        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x  = self.cnn1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.cnn2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.cnn3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.cnn4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(-1, 320)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x
    
def train(model, train_loader, val_loader,criterion, optimizer, num_epochs=5,patience=3):

    history = {'train_loss': [], 'val_acc': []}
    best_epoch = 0
    best_acc_in_val = 0.0
    
    for epoch in range(num_epochs):
        print(f"开始第 {epoch+1} 轮训练")
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        end_time = time.time()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Time: {end_time - start_time:.2f}s")

        print(f"开始第{epoch+1}轮验证")
        with torch.no_grad():
            model.eval()
            total = 0
            correct = 0
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print(f'验证集正确率: {accuracy:.2f}%')

            if accuracy > best_acc_in_val:
                best_acc_in_val = accuracy
                best_epoch = epoch
                torch.save(model.state_dict(), "./models/best_mnist_classfier_plus.pth")
                print("破纪录,当前最佳模型参数。")

        history['train_loss'].append(running_loss/len(train_loader))
        history['val_acc'].append(accuracy)

        if epoch - best_epoch >= patience:
            print("验证集准确率连续多轮未提升，提前停止训练。")
            break

    print("训练完成.")
    return history


def acc_in_test(model, test_loader):
    
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'测试集正确率: {accuracy:.2f}%')
    return accuracy

    
    

if __name__ == "__main__":
    #超参数
    #########################################
    lr=0.001
    num_epochs=100
    patience=10
    weight_decay=1e-4
    
    #########################################


    train_loader ,val_loader= load_MNIST(isTrain=True)
    test_loader = load_MNIST(isTrain=False)

    model = MnistClassifierPlus()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = train(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, patience=patience)
    test_accuracy = acc_in_test(model, test_loader)




    


  

  