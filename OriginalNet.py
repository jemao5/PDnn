import torch
import torch.nn as nn
from torch.utils import data
from data_functions import load_images2, one_hot_labels
import numpy as np
import DataHandler
from tqdm import tqdm
import cnn
import matplotlib.pyplot as plt

data_path = r"C:\Users\jonat\Documents\InspiritAI\PDdata"

spiral_test_healthy = load_images2((data_path + "\\spiral\\testing\\healthy"), (256, 256))
spiral_test_parkinson = load_images2((data_path + "\\spiral\\testing\\parkinson"), (256, 256))
spiral_test_images = np.concatenate((spiral_test_parkinson, spiral_test_healthy), axis=0)
spiral_test_labels = np.concatenate(([0] * spiral_test_healthy.shape[2], [1] * spiral_test_parkinson.shape[2]), axis=0)


spiral_train_healthy = load_images2((data_path + "\\spiral\\training\\healthy"), (256, 256))
spiral_train_parkinson = load_images2((data_path + "\\spiral\\training\\parkinson"), (256, 256))
spiral_train_images = np.concatenate((spiral_train_parkinson, spiral_train_healthy), axis=0)
spiral_train_labels = np.concatenate(([0] * spiral_train_healthy.shape[0], [1] * spiral_train_parkinson.shape[0]), axis=0)


# Define relevant variables for the ML task
batch_size = 24
learning_rate = 0.001
num_epochs = 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = cnn.CNN(device)
optimizer = torch.optim.Adam(model.net.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

spiral_train_labels = one_hot_labels(spiral_train_labels)
spiral_test_labels = one_hot_labels(spiral_test_labels)

train_dataset = DataHandler.DataHandler(spiral_train_images, spiral_train_labels)
test_dataset = DataHandler.DataHandler(spiral_test_images, spiral_test_labels)
train_dataloader = data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
test_dataloader = data.DataLoader(test_dataset, batch_size, shuffle=True, num_workers=0)

losses = []
for epoch in tqdm(range(num_epochs)):

    for (label, image) in train_dataloader:
        prediction = model.forward(image.to(device=device, dtype=torch.float))
        # print(prediction)
        # print(label)
        loss = loss_function(prediction.to(device), label.to(device=device, dtype=torch.float))


        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        loss.backward()
        optimizer.step()

    losses.append(loss.item())

    # with torch.no_grad()
    #     for b,(test_image, test_label) in enumerate()

print(losses)
plt.plot(losses)
plt.show()


