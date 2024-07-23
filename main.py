import torch
from yolo_implementation import yolov1, CustomLoss
from generate_data import ImageDataset
from torch.utils.data import DataLoader
import torch.optim as optim

if __name__=="__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = yolov1().to(device)
    image_data = ImageDataset()
    criterion = CustomLoss()
    image_loader = DataLoader(dataset=image_data, batch_size=1)
    optimizer = optim.SGD(model.parameters(), lr=0.0025)
    EPOCHS = 2
    for epoch in range(EPOCHS):
        running_loss = 0
        for inputs, labels in image_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: Loss = {running_loss}")