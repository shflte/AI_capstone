import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

class CaptchaDataset(Dataset):
    def __init__(self, data, root, return_filename=False):
        self.data = [sample for sample in data]
        self.return_filename = return_filename
        self.root = root

    def __getitem__(self, index):
        filename, label = self.data[index]

        target_size = (125, 68)
        resize_transform = transforms.Resize(target_size, antialias=None)
        img = torchvision.io.read_image(f"{self.root}/{filename}")
        img = resize_transform(img)
        img_pil = transforms.ToPILImage()(img)

        transform = transforms.Compose([
            transforms.ToPILImage(), 
            transforms.Resize(288),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = transform(img)

        if self.return_filename:
            return torch.FloatTensor(img / 255), filename
        else:

            table = ['0','1','2','3','4','5','6','7','8','9']
            one_hot = [0]*40
            index0 = table.index(label[0])
            index1 = table.index(label[1])+10
            index2 = table.index(label[2])+20
            index3 = table.index(label[3])+30
            one_hot[index0] = 1
            one_hot[index1] = 1
            one_hot[index2] = 1
            one_hot[index3] = 1
            return torch.FloatTensor(img / 255), torch.LongTensor(one_hot)

    def __len__(self):
        return len(self.data)


def get_captcha_value():
    TEST_PATH = "."
    device = "cpu"
    IMAGE_NAME = "image.png"
    
    test_data = [[IMAGE_NAME, '0']]
    test_ds = CaptchaDataset(test_data, root=TEST_PATH, return_filename=True)
    test_dl = DataLoader(test_ds, batch_size=32, num_workers=4, drop_last=False, shuffle=False)

    model = torchvision.models.resnet18(weights=None).to(device)
    model.fc = nn.Linear(in_features=512, out_features=40, bias=True).to(device)
    model.load_state_dict(torch.load("models/optimal_model", map_location=torch.device('cpu')))
    model.eval()

    table = ['0','1','2','3','4','5','6','7','8','9']
    for image, filenames in test_dl:
        image = image.to(device)
        pred = model(image)
        for i in range(len(filenames)):
            ans = ""
            large1_pred = torch.argmax(pred[i][0:10])
            large2_pred = torch.argmax(pred[i][10:20])
            large3_pred = torch.argmax(pred[i][20:30])
            large4_pred = torch.argmax(pred[i][30:40])
            ans = ans + table[large1_pred]
            ans = ans + table[large2_pred]
            ans = ans + table[large3_pred]
            ans = ans + table[large4_pred]
            return int(ans)
        del image, filenames

print(get_captcha_value())
