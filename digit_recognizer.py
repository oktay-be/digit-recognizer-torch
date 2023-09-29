# import mlflow.pytorch
# import torch
# Define your PyTorch model
# model = torch.nn.Sequential(
#     torch.nn.Linear(2, 1),
#     torch.nn.Sigmoid()
# )
# # Train your model and obtain the trained model object
# # Log the model to MLflow
# mlflow.pytorch.log_model(
#     pytorch_model=model,
#     artifact_path="my-model",
#     conda_env="path/to/conda.yaml",
#     code_paths=["path/to/training/script.py", "path/to/other/code"],
#     registered_model_name="my-registered-model"
# )
# # Save the model to a file
# mlflow.pytorch.save_model(
#     pytorch_model=model,
#     path="my-model",
#     conda_env="path/to/conda.yaml",
#     code_paths=["path/to/training/script.py", "path/to/other/code"]
# )

import torch
import torchvision

n_epochs = 1
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

example_data.shape

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), './results/model.pth')
      torch.save(optimizer.state_dict(), './results/optimizer.pth')
      
      
def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  accuracy = 100. * correct / len(test_loader.dataset)
  accuracy = accuracy.numpy()
  accuracy = accuracy.flat[0]
  return accuracy
  
test()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  accuracy = test()
   
print("Accuracy:")
print(accuracy)

import mlflow

with mlflow.start_run() as run:
    #mlflow.pytorch.log_model(model, "pytorch_model")
    # mlflow.tracking.MlflowClient.create_registered_model
    # run.info.lifecycle_stage
    model_state = torch.load('./results/model.pth')

    registered_model = Net()  # Create your model class

    # Load the state into the registered model
    registered_model.load_state_dict(model_state)

    # Calc metrics
    acc = accuracy
    
    # Print metrics
    print("  acc: {}".format(acc))

    # Log metrics
    mlflow.log_metric("acc", acc)

    mlflow.set_tag('candidate', 'true')

    run_id = mlflow.active_run().info.run_uuid
    print(run_id)
    
    print("inter")

# Log the model to MLflow
    mlflow.pytorch.log_model(
        pytorch_model=registered_model,
        artifact_path="my-model",
        #conda_env="C:\Users\A00008198\AppData\Local\miniconda3\envs\digit_recognizer",
        #code_paths=["path/to/training/script.py", "path/to/other/code"],
        registered_model_name="my_registered_pytorch_model",
    )



    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    model = client.get_latest_versions("my_registered_pytorch_model", stages=['None'])
    versions = model[0]
    version = versions.version
    
    
    print("Last Production Model versions:", versions)
    client.transition_model_version_stage(
    name="my_registered_pytorch_model",
    version=version,
    stage="Staging",
    archive_existing_versions=True
    )
# mlflow.pytorch.log_model(registered_model, 'my_registered_pytorch_model')

# with mlflow.start_run():
#     # Log the model using MLflow
#     mlflow.pytorch.log_model(registered_model, 'my_registered_pytorch_model')

print("end")