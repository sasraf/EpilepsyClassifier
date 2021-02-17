import pickle
import torch
import time
import numpy as np
import sklearn.model_selection

from tester import accuracy, generateROC

# Load data
data = pickle.load(open('seizureDataSerialized.txt', "rb"))
inputData = np.array(data['inputs'])
expectedOutputs = np.array(data['outputs'])

# train with 80% of data, test with 20%: 400 and 100 samples respectively
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(inputData, expectedOutputs, test_size=.2)

# Configure model architecture
model = torch.nn.Sequential(
    torch.nn.Conv1d(1, 5, 3),
    torch.nn.MaxPool1d(2),
    torch.nn.LeakyReLU(),
    torch.nn.Conv1d(5, 10, 7),
    torch.nn.MaxPool1d(2),
    torch.nn.LeakyReLU(),
    torch.nn.Conv1d(10, 1, 9),
    torch.nn.MaxPool1d(3),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(337, 30),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(30, 2),
    torch.nn.Sigmoid()
)

# Initialize model loss, learning rate, epoch, optimizer
lossFunction = torch.nn.MSELoss()
epochs = 100
learningRate = .001
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

start = time.time()

# convert data to tensors
x_train = torch.tensor(x_train).unsqueeze(1).float()
y_train = torch.tensor(y_train).float()

# Train model
for i in range(epochs):

    # Predict output from inputdata
    predictedOutput = model(x_train).squeeze(1)

    # Calculate loss
    loss = lossFunction(predictedOutput, y_train)

    # Print t and loss for debugging purposes
    if i % 5 == 0:
        print(i, loss.item())

    # Zero gradients before backwards pass
    optimizer.zero_grad()

    # Compute gradient of loss with respect to parameters
    loss.backward()

    # Update weights
    optimizer.step()

end = time.time()

print("Time: " + str(end - start) + " seconds")

# Test accuracy of model over train and test data to determine over/underfitting
accuracy(x_train, y_train, model)
x_test = torch.tensor(x_test).float().unsqueeze(1)
accuracy(x_test, y_test, model)

# Generate ROC
generateROC(x_test, y_test, model)

# Decide whether or not to save model
yesno = input("Save model? \"y\" to save")
if yesno == "y":
    pickle.dump(model, open("models/model.txt", "wb"))

exit(0)

