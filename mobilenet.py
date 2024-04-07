from typing import List, Dict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd

# ENABLING GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# CUSTOM FUNCTION TO TRAIN MODEL
def train(model: nn.Module,
          loss_fn: nn.modules.loss._Loss,
          optimizer: torch.optim.Optimizer,
          train_loader: torch.utils.data.DataLoader,
          epoch: int=0)-> List:

    train_loss = []
    model.train()

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, targets) 
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if batch_idx % (len(train_loader) // 8) == 0: # We visulize our output 10 times.
            print(f'Epoch {epoch}: [{batch_idx*len(images)}/{len(train_loader.dataset)}] Loss: {loss.item():.3f}')

    assert len(train_loss) == len(train_loader)
    return train_loss

# CUSTOM FUNCTION TO TEST MODEL
def test(model: nn.Module,
         loss_fn: nn.modules.loss._Loss,
         test_loader: torch.utils.data.DataLoader,
         epoch: int=0)-> Dict:

    model.eval() # we need to set the mode for our model
    test_loss = 0
    test_stat = {}
    prediction = torch.zeros(0, dtype=torch.long)
    correct = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            output = model(images)
            test_loss += len(targets)*loss_fn(output, targets).item()
            pred = output.data.argmax(1) # getting the largest class value
            correct += pred.eq(targets).sum() # sum up the correct pred
            prediction = torch.cat((prediction, pred.to('cpu')), dim=0)

    total_num = len(test_loader.dataset)

    # STORING AND DISPLAYING RESULTS
    test_stat = {"loss": test_loss / total_num, "accuracy": correct / total_num, "prediction": prediction}
    print(f"Test result on epoch {epoch}: total sample: {total_num}, Avg loss: {test_stat['loss']:.3f}, Acc: {100*test_stat['accuracy']:.3f}%")
    
    # ASSERTIONS AS SANITY CHECKS
    assert "loss" and "accuracy" and "prediction" in test_stat.keys()
    assert len(test_stat["prediction"]) == len(test_loader.dataset)
    assert isinstance(test_stat["prediction"], torch.Tensor)
    return test_stat

# FUNCTION TO MAKE OUTPUT FILE
def make_file(model: nn.Module, path):
    all_files = set(range(0,4977))
    classified_files = set()
    file_df = pd.DataFrame(columns=['Id', 'Category'])

    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        
        if os.path.isfile(f):
            img = Image.open(f)
            #img = cv2.imread(f)
            transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()])
            img = transform(img)
            img = img.unsqueeze(0)
            img = img.to(device)
            output = model(img)
            pred = output.data.argmax(1)
            id = int(filename[:-4])
            idx = pred.to('cpu')[0].item()
            classified_files.add(id)
            file_df.loc[len(file_df.index)] = [id, class_labels[idx]]

    for i in (all_files - classified_files):
        file_df.loc[len(file_df.index)] = [i, class_labels[46]] 

    file_df = file_df.sort_values(by='Id')
    file_df.to_csv('out.csv', index=False) 


# PATH TO IMAGE SET FOR TRAINING
base_dir = 'celeb_folder_cropped_full'

##### Mobilenet Classifier #####
mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
mobilenet.classifier[1] = nn.Linear(1280,100) # Changing output to only 100 classes
mobilenet = mobilenet.to(device)

transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor()])

# DATA SPLITTING
all_data_object = torchvision.datasets.ImageFolder(root=base_dir, transform=transform) 
train_size = int(0.98 * len(all_data_object))
test_size = len(all_data_object) - train_size
train_data_object, test_data_object = torch.utils.data.random_split(all_data_object, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_data_object, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data_object, batch_size=32, shuffle=False)

class_labels = train_data_object.dataset.classes # Getting labels 

# TRAINING MODEL
criterion = nn.CrossEntropyLoss()
num_epoch = 18
optimizer = optim.Adam(mobilenet.parameters(), lr=0.001)

for epoch in range(1, num_epoch+1):  # loop over the dataset multiple times
    train(mobilenet, criterion, optimizer, train_loader, epoch)

retval = test(mobilenet, criterion, test_loader)

make_file(mobilenet, 'cropped_test')

print(retval['accuracy'])