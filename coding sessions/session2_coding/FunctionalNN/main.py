from Dataset import load_date
from Model import Network
from helper_functions import train

#Loading data
train_loader, validation_loader = load_date()
#creating a model
model = Network()
# training the model
train(model,train_loader,validation_loader)




