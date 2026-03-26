import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

class CNN_MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64*7*7, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

class CNN_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64*8*8, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def fgsm_targeted(model, x, target, eps=0.1): # target은 target label을 의미
    x.requires_grad_(True) # image에 기울기 설정
    output = model(x)
    loss = F.cross_entropy(output, target)
    
    model.zero_grad() # 기울기는 누적될 필요가 없음 -> 0
    loss.backward()

    x_adv = x - eps * x.grad.detach().sign() # 기울기를 연산 기록에서 안전하게 분리
    x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv


def fgsm_untargeted(model, x, label, eps=0.1): # correct label
    x.requires_grad_(True)
    output = model(x)
    loss = F.cross_entropy(output, label)

    model.zero_grad()
    loss.backward()
    x_adv = x + eps * x.grad.detach().sign()
    x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv


def pgd_targeted(model, x, target, k=40, eps=0.3, eps_step=0.01):
    x_adv = x.clone().detach()
    for i in range(k):
        x_adv.requires_grad_(True)
        output = model(x_adv)
        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()
        x_adv = x_adv - eps_step * x_adv.grad.detach().sign()
        x_adv = torch.clamp(x_adv, x-eps, x+eps)
        x_adv = torch.clamp(x_adv, 0, 1)
        x_adv = x_adv.detach()
    return x_adv


def pgd_untargeted(model, x, label, k=40, eps=0.3, eps_step=0.01): # correct label
    x_adv = x.clone().detach()
    for i in range(k):
        x_adv.requires_grad_(True)
        output = model(x_adv)
        loss = F.cross_entropy(output, label)
        model.zero_grad()
        loss.backward()
        x_adv = x_adv + eps_step * x_adv.grad.detach().sign()
        x_adv = torch.clamp(x_adv, x-eps, x+eps)
        x_adv = torch.clamp(x_adv, 0, 1)
        x_adv = x_adv.detach()
    return x_adv

# 결과 이미지를 저장할 디렉토리 생성
os.makedirs("results", exist_ok=True)

# 하이퍼파라미터 및 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
epochs = 30
epsilons = [0.05, 0.1, 0.2, 0.3] # 레포트 작성을 위한 다양한 노이즈 크기
num_test_samples = 100 # 평가할 샘플 수

def get_dataloaders(dataset_name):
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset_name == 'MNIST':
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else: # CIFAR10
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    # 공격 시각화를 쉽게 추출하기 위해 testloader는 batch_size=1로 설정
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False) 
    return trainloader, testloader

def train_model(model, trainloader, dataset_name):
    print(f"\n--- Training {dataset_name} Model ---")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(trainloader):.4f}")
    return model

def save_visualization(orig_img, adv_img, orig_pred, adv_pred, filename):
    # 텐서를 넘파이 배열로 변환 (채널 위치 조정)
    orig_img = orig_img.squeeze().detach().cpu().numpy()
    adv_img = adv_img.squeeze().detach().cpu().numpy()
    
    perturbation = abs(adv_img - orig_img)
    perturbation = perturbation * 10 # 가시성을 위해 10배 증폭
    perturbation = perturbation.clip(0, 1) # 0~1 사이로 제한

    # RGB 이미지 처리
    if len(orig_img.shape) == 3:
        orig_img = orig_img.transpose(1, 2, 0)
        adv_img = adv_img.transpose(1, 2, 0)
        perturbation = perturbation.transpose(1, 2, 0)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(orig_img, cmap='gray' if len(orig_img.shape)==2 else None)
    axes[0].set_title(f"Original\nPred: {orig_pred}")
    axes[0].axis('off')

    axes[1].imshow(adv_img, cmap='gray' if len(adv_img.shape)==2 else None)
    axes[1].set_title(f"Adversarial\nPred: {adv_pred}")
    axes[1].axis('off')

    axes[2].imshow(perturbation, cmap='gray' if len(perturbation.shape)==2 else None)
    axes[2].set_title("Perturbation\n(Magnified)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f"results/{filename}")
    plt.close()

def evaluate_attacks(model, testloader, dataset_name):
    model.eval()
    attack_funcs = {
        'FGSM Targeted': fgsm_targeted,
        'FGSM Untargeted': fgsm_untargeted,
        'PGD Targeted': pgd_targeted,
        'PGD Untargeted': pgd_untargeted
    }

    for attack_name, attack_func in attack_funcs.items():
        print(f"\n[ Running {attack_name} on {dataset_name} ]")
        is_targeted = 'Targeted' in attack_name
        
        for eps in epsilons:
            successful_attacks = 0
            total_samples = 0
            saved_images = 0
            
            for images, labels in testloader:
                if total_samples >= num_test_samples:
                    break
                    
                images, labels = images.to(device), labels.to(device)
                
                # 원본 이미지가 올바르게 분류되는지 먼저 확인 (틀린 문제는 공격 제외)
                with torch.no_grad():
                    orig_outputs = model(images)
                orig_pred = orig_outputs.argmax(dim=1)
                if orig_pred.item() != labels.item():
                    continue 

                # 타겟 설정: 정답 클래스가 아닌 다른 클래스 (예: label + 1)
                target_labels = (labels + 1) % 10 if is_targeted else labels

                # 공격 수행
                if is_targeted:
                    adv_images = attack_func(model, images, target_labels, eps=eps)
                else:
                    adv_images = attack_func(model, images, labels, eps=eps)

                # 공격받은 이미지의 예측 결과
                with torch.no_grad():
                    adv_outputs = model(adv_images)
                adv_pred = adv_outputs.argmax(dim=1)

                # 성공 여부 판별
                if is_targeted:
                    success = (adv_pred.item() == target_labels.item())
                else:
                    success = (adv_pred.item() != labels.item())

                if success:
                    successful_attacks += 1
                    # epsilon=0.3일 때 공격 성공 샘플 5장 시각화 저장
                    if eps == 0.3 and saved_images < 5: 
                        filename = f"{dataset_name}_{attack_name.replace(' ', '_')}_eps{eps}_sample{saved_images+1}.png"
                        save_visualization(images, adv_images, orig_pred.item(), adv_pred.item(), filename)
                        saved_images += 1
                        
                total_samples += 1

            asr = (successful_attacks / total_samples) * 100 if total_samples > 0 else 0
            print(f"Epsilon: {eps:.2f} | Attack Success Rate: {asr:.1f}%")

def main():
    datasets = ['MNIST', 'CIFAR10']
    
    for dataset in datasets:
        trainloader, testloader = get_dataloaders(dataset)
        
        # 모델 선택 및 학습
        model = CNN_MNIST() if dataset == 'MNIST' else CNN_CIFAR10()
        trained_model = train_model(model, trainloader, dataset)
        
        # 공격 평가 및 결과 저장
        evaluate_attacks(trained_model, testloader, dataset)

if __name__ == '__main__':
    main()