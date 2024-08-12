import numpy as np
import csv
import json
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Transformer
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_squared_error, r2_score

torch.manual_seed(42)

y_type_mapping = {
    "5_YR_R": 0,
    "7_YR_R": 1,
    "10_YR_R": 2
}

class Dense_NN_3layers(nn.Module):
    def __init__(self, net_params):
        super(Dense_NN_3layers, self).__init__()

        self.fc1 = nn.Linear(net_params[0],  net_params[1])
        self.fc2 = nn.Linear(net_params[1],  net_params[2])
        self.fc3 = nn.Linear(net_params[2],  net_params[3])
        self.fc4 = nn.Linear(net_params[3], 1)  # Output layer with one float output

    def forward(self, input):
        # Fully connected layer F1: (input, net_params[1])
        f1 = F.relu(self.fc1(input))
        # Fully connected layer F2: (net_params[1], net_params[2])
        f2 = F.relu(self.fc2(f1))
        # Fully connected layer F3: (net_params[2], net_params[3])
        f3 = F.relu(self.fc3(f2))

        output = self.fc4(f3)

        return output


class Dense_NN_3layers_Dropout(nn.Module):
    def __init__(self, net_params):
        super(Dense_NN_3layers_Dropout, self).__init__()

        self.fc1 = nn.Linear(net_params[0], net_params[1])
        self.fc2 = nn.Linear(net_params[1], net_params[2])
        self.fc3 = nn.Linear(net_params[2], net_params[3])
        self.fc4 = nn.Linear(net_params[3], 1)  # Output layer with one float output

        # Define dropout layers with a probability of 0.5
        self.dropout = nn.Dropout(0.75)

    def forward(self, input):
        # Fully connected layer F1 with ReLU and dropout
        f1 = F.relu(self.fc1(input))
        f1 = self.dropout(f1)

        # Fully connected layer F2 with ReLU and dropout
        f2 = F.relu(self.fc2(f1))
        f2 = self.dropout(f2)

        # Fully connected layer F3 with ReLU and dropout
        f3 = F.relu(self.fc3(f2))
        f3 = self.dropout(f3)

        output = self.fc4(f3)

        return output


class LSTM_Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_prob=0.5):
        super(LSTM_Network, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def Dense_NN_3layers_model_search(x_train, x_valid, x_train_valid, x_test, y_train, y_valid, y_train_valid, y_test, y_types):
	print('-=Dense NN (3 layers)=-')

	for y_type in y_types:

		y_type_id = y_type_mapping.get(y_type, None)  # Use get with default value None
		if y_type_id is None:
			print(f"Incorrect y_type input: {y_type}")
			break

		layer1_dims = [50] #[50, 100, 200, 300, 400, 500]  # Example dimensions for the first layer
		layer2_dims = [50] #[25, 50, 100, 150, 200, 300]    # Example dimensions for the second layer
		layer3_dims = [10] #[5, 10, 25, 50, 100]    # Example dimensions for the third layer

		num_epochs = 8
		learning_rate=0.0001 
		batch_size=1

		best_val_loss = np.inf
		best_config = None

		history_train = np.zeros((len(layer1_dims), len(layer2_dims), len(layer3_dims)))
		history_valid = np.zeros((len(layer1_dims), len(layer2_dims), len(layer3_dims)))
		best_val_r_sq = 0

		x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
		x_valid_tensor = torch.tensor(x_valid, dtype=torch.float32)
		x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
		y_train_tensor = torch.tensor(y_train[:, y_type_id], dtype=torch.float32) 
		y_valid_tensor = torch.tensor(y_valid[:, y_type_id], dtype=torch.float32) 

		#print(f'The datatype of x_train_tensor: {x_train_tensor.dtype}\n')
		#print(f'The shape of x_train_tensor: {x_train_tensor.shape}\n')

		for i, layer1_dim in enumerate(layer1_dims):
			for j, layer2_dim in enumerate(layer2_dims):
				for k, layer3_dim in enumerate(layer3_dims):
					print(f"Training model with layer1_dim: {layer1_dim}, layer2_dim: {layer2_dim}, layer3_dim: {layer3_dim}")
					net_params = [x_train.shape[1], layer1_dim, layer2_dim, layer3_dim]
					model = Dense_NN_3layers(net_params)
					#print(model)

					criterion = nn.MSELoss()
					optimizer = optim.Adam(model.parameters(), lr=learning_rate)

					dataset = TensorDataset(x_train_tensor, y_train_tensor)
					dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

					dataset_valid = TensorDataset(x_valid_tensor, y_valid_tensor)

					loss_train_history = []
					loss_valid_history	= []
					for epoch in range(num_epochs):
						for batch_idx, (data, target) in enumerate(dataloader):
							# Zero the gradients
							optimizer.zero_grad()
							# Forward pass
							outputs = model(data)
							target = target.view(-1, 1)
							# Compute the loss
							loss = criterion(outputs, target)
							# Backward pass
							loss.backward()
							# Update the weights
							optimizer.step()
						model.eval()
						print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

					# Check the best validation r_sq
					y_pred_train = model(x_train_tensor).detach().numpy()
					y_pred_valid = model(x_valid_tensor).detach().numpy()
					y_pred_test = model(x_test_tensor).detach().numpy()

					history_train[i, j, k] = r2_score(y_train[:, y_type_id], y_pred_train)
					val_r_sq = r2_score(y_valid[:, y_type_id], y_pred_valid)
					history_valid[i, j, k] = val_r_sq

					if val_r_sq > best_val_r_sq:
						best_val_r_sq = val_r_sq
						best_config = (layer1_dim, layer2_dim, layer3_dim)
						print(f'New best config: {best_config} with r_sq: {best_val_r_sq:.3f}')
							
					print(
					    f'Dense NN ({y_type}) for train set: '
					    f'MSE = {mean_squared_error(y_train[:, y_type_id], y_pred_train):.3f} and '
					    f'R-squared = {r2_score(y_train[:, y_type_id], y_pred_train):.3f}'
					    )

					print(
					    f'Dense NN ({y_type}) for test set: '
					    f'MSE = {mean_squared_error(y_test[:, y_type_id], y_pred_test):.3f} and '
					    f'R-squared = {r2_score(y_test[:, y_type_id], y_pred_test):.3f}\n'
					    ) 

		#print(history_valid)
		print(f'Best configuration: {best_config[0],best_config[1],best_config[2]}')

		# Convert the 3D array to a nested list
		history_valid_list = history_valid.tolist()

		# Save to JSON
		with open('history_valid.json', 'w') as jsonfile:
			json.dump(history_valid_list, jsonfile)


def Dense_NN_3layers_model_optimization(x_train, x_valid, x_train_valid, x_test, 
										y_train, y_valid, y_train_valid, y_test, y_types):
	print('-=Dense NN (3 layers)=-')

	for y_type in y_types:

		y_type_id = y_type_mapping.get(y_type, None)  # Use get with default value None
		if y_type_id is None:
			print(f"Incorrect y_type input: {y_type}")
			break

		layer1_dim = 50
		layer2_dim = 50
		layer3_dim = 10

		num_epochs = 10
		learning_rate=0.0001 
		batch_size=1

		x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
		x_valid_tensor = torch.tensor(x_valid, dtype=torch.float32)
		x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
		y_train_tensor = torch.tensor(y_train[:, y_type_id], dtype=torch.float32) 
		y_valid_tensor = torch.tensor(y_valid[:, y_type_id], dtype=torch.float32) 

		print(f"Training model with layer1_dim: {layer1_dim}, layer2_dim: {layer2_dim}, layer3_dim: {layer3_dim}")
		net_params = [x_train.shape[1], layer1_dim, layer2_dim, layer3_dim]
		model = Dense_NN_3layers(net_params)

		criterion = nn.MSELoss()
		optimizer = optim.Adam(model.parameters(), lr=learning_rate)

		dataset = TensorDataset(x_train_tensor, y_train_tensor)
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

		dataset_valid = TensorDataset(x_valid_tensor, y_valid_tensor)

		loss_train_history = []
		loss_valid_history	= []
		for epoch in range(num_epochs):
			for batch_idx, (data, target) in enumerate(dataloader):
				# Zero the gradients
				optimizer.zero_grad()
				# Forward pass
				outputs = model(data)
				target = target.view(-1, 1)
				# Compute the loss
				loss = criterion(outputs, target)
				# Backward pass
				loss.backward()
				# Update the weights
				optimizer.step()
			print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

			model.eval()
			with torch.no_grad():
				loss_train = criterion(model(x_train_tensor), y_train_tensor.view(-1, 1))
				loss_train_history.append(loss_train.numpy())

				loss_valid = criterion(model(x_valid_tensor), y_valid_tensor.view(-1, 1))
				loss_valid_history.append(loss_valid.numpy())

		formatted_train_history = [f"{x:.3f}" for x in loss_train_history]
		formatted_valid_history = [f"{x:.3f}" for x in loss_valid_history]
		print(f"Training r_sq history: {formatted_train_history}")
		print(f"Validation r_sq history: {formatted_valid_history}")

		if True:
			epochs = range(1, num_epochs+1)
			plt.figure(figsize=(10, 5))
			plt.plot(epochs, loss_train_history, label='Training R^2', marker='o')
			plt.plot(epochs, loss_valid_history, label='Validation R^2', marker='o')
			plt.xlabel('Epochs')
			plt.ylabel('R^2 Score')
			plt.title('Training and Validation R^2 Score Over Epochs')
			plt.legend()
			plt.grid(True)
			plt.show()
			### 

		# Check the best validation r_sq
		y_pred_train = model(x_train_tensor).detach().numpy()
		y_pred_test = model(x_test_tensor).detach().numpy()
		
		print(
		    f'Dense NN ({y_type}) for train set: '
		    f'MSE = {mean_squared_error(y_train[:, y_type_id], y_pred_train):.3f} and '
		    f'R-squared = {r2_score(y_train[:, y_type_id], y_pred_train):.3f}'
		    )

		print(
		    f'Dense NN ({y_type}) for test set: '
		    f'MSE = {mean_squared_error(y_test[:, y_type_id], y_pred_test):.3f} and '
		    f'R-squared = {r2_score(y_test[:, y_type_id], y_pred_test):.3f}\n'
		    ) 

def prepare_lstm_dataset(x_data, y_data, sequence_length):
    """
    Prepares the dataset for LSTM training by creating sequences of the specified length.
    """
    x_sequences = []
    y_sequences = []
    for i in range(len(x_data) - sequence_length + 1):
        x_seq = x_data[i:i + sequence_length]
        y_seq = y_data[i + sequence_length - 1]  # Target is the value after the end of the sequence
        x_sequences.append(x_seq)
        y_sequences.append(y_seq)
    
    x_array = np.array(x_sequences, dtype=np.float32)
    y_array = np.array(y_sequences, dtype=np.float32).reshape(-1, 1)
    
    x_tensor = torch.tensor(x_array)
    y_tensor = torch.tensor(y_array)
    return x_tensor, y_tensor


def LSTM_model_optimization(x_train, x_valid, x_train_valid, x_test, 
							y_train, y_valid, y_train_valid, y_test, y_types, sequence_length=5):
    print('-=LSTM Model Optimization=-')

    for y_type in y_types:
        y_type_id = y_type_mapping.get(y_type, None)
        if y_type_id is None:
            print(f"Incorrect y_type input: {y_type}")
            break

        input_dim = x_train.shape[1]  # Number of features
        hidden_dim = 75
        output_dim = 1  # Predicting one value
        num_layers = 2
        dropout_prob = 0.5
        num_epochs = 15
        learning_rate = 0.001
        batch_size = 32

        # Prepare datasets
        x_train_tensor, y_train_tensor = prepare_lstm_dataset(x_train, y_train[:, y_type_id], sequence_length)
        x_valid_tensor, y_valid_tensor = prepare_lstm_dataset(x_valid, y_valid[:, y_type_id], sequence_length)
        x_test_tensor, y_test_tensor = prepare_lstm_dataset(x_test, y_test[:, y_type_id], sequence_length)

        print(f"Training LSTM model with input_dim: {input_dim}, hidden_dim: {hidden_dim}, num_layers: {num_layers}")
        model = LSTM_Network(input_dim, hidden_dim, output_dim, num_layers, dropout_prob)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Creating dataset and dataloader with the correct shape
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        valid_dataset = TensorDataset(x_valid_tensor, y_valid_tensor)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        loss_train_history = []
        loss_valid_history = []
        for epoch in range(num_epochs):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                train_preds = model(x_train_tensor)
                valid_preds = model(x_valid_tensor)
                loss_train = criterion(train_preds, y_train_tensor)
                loss_valid = criterion(valid_preds, y_valid_tensor)
                loss_train_history.append(loss_train.item())
                loss_valid_history.append(loss_valid.item())

            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss_train.item():.4f}, Valid Loss: {loss_valid.item():.4f}')

        formatted_train_history = [f"{x:.3f}" for x in loss_train_history]
        formatted_valid_history = [f"{x:.3f}" for x in loss_valid_history]
        print(f"Training loss history: {formatted_train_history}")
        print(f"Validation loss history: {formatted_valid_history}")

        if False: # graph print
        	epochs = range(1, num_epochs+1)
        	plt.figure(figsize=(10, 5))
        	plt.plot(epochs, loss_train_history, label='Training R^2', marker='o')
        	plt.plot(epochs, loss_valid_history, label='Validation R^2', marker='o')
        	plt.xlabel('Epochs')
        	plt.ylabel('R^2 Score')
        	plt.title('Training and Validation R^2 Score Over Epochs')
        	plt.legend()
        	plt.grid(True)
        	plt.show()
			### 

        # Evaluate on train and test sets
        model.eval()
        with torch.no_grad():
            y_pred_train = model(x_train_tensor).detach().numpy()
            y_pred_test = model(x_test_tensor).detach().numpy()

        # Align y_train and y_test with sequence length
        y_train_aligned = y_train[sequence_length - 1:, y_type_id].flatten()
        y_test_aligned = y_test[sequence_length - 1:, y_type_id].flatten()

        # Flatten the predictions to 1D array for comparison
        y_pred_train_flat = y_pred_train.flatten()
        y_pred_test_flat = y_pred_test.flatten()

        print(
            f'LSTM ({y_type}) for train set: '
            f'MSE = {mean_squared_error(y_train_aligned, y_pred_train_flat):.3f} and '
            f'R-squared = {r2_score(y_train_aligned, y_pred_train_flat):.3f}'
        )

        print(
            f'LSTM ({y_type}) for test set: '
            f'MSE = {mean_squared_error(y_test_aligned, y_pred_test_flat):.3f} and '
            f'R-squared = {r2_score(y_test_aligned, y_pred_test_flat):.3f}\n'
        )

################# 	
