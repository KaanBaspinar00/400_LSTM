import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import READ_FILES
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error

# GET DATA
Reflectance_Data = READ_FILES.result_ref
Thickness_Data = READ_FILES.result_thick
MaterialData = READ_FILES.result_material
number_of_layers = 12
Thickness_Data_padded = []

for i in Thickness_Data:
    padded = i + [0] * (number_of_layers - len(i))
    Thickness_Data_padded.append(np.array(padded))

Thickness_Data_padded = np.array(Thickness_Data_padded)

MaterialData_padded = []

for sublist in MaterialData:
    # Calculate the number of padding rows required
    padding_rows = number_of_layers - len(sublist)

    # Pad the sublist with zeros
    padded_sublist = np.pad(sublist, ((0, padding_rows), (0, 0)), mode='constant')

    # Append the padded sublist to the new list
    MaterialData_padded.append(padded_sublist)

# Now our data : MaterialData_padded ; Thickness_Data_padded ; Reflectance_Data

thicknesses = np.array(Thickness_Data_padded)
materials_data = np.array(MaterialData_padded)
reflectance = np.array(Reflectance_Data)

materials_data_reshaped = materials_data


# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, output_size, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads,
                                                                    dim_feedforward=hidden_size, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(input_size, output_size)

    def forward(self, reflectance, materials):
        reflectance = reflectance.unsqueeze(0)  # Add a batch dimension
        output = self.transformer_encoder(reflectance)
        output = self.decoder(output.squeeze(0))
        return output


# Convert data to PyTorch tensors
reflectance_tensor = torch.tensor(reflectance, dtype=torch.float32)
materials_tensor = torch.tensor(materials_data_reshaped, dtype=torch.float32)
thickness_tensor = torch.tensor(thicknesses, dtype=torch.float32)

# Split data into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(reflectance_tensor, thickness_tensor, test_size=0.20,
                                                    random_state=123)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=123)

# Create DataLoader for train, validation, and test sets
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

# Define hyperparameters
input_size = 200
hidden_size = 128
num_layers = 4
num_heads = 4
output_size = 12
dropout = 0.01

"""
# Initialize the model
model = TransformerModel(input_size, hidden_size, num_layers, num_heads, output_size, dropout=dropout)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)


# Training loop
def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for reflectance, thickness in train_loader:
            optimizer.zero_grad()
            output = model(reflectance, materials_tensor)  # Pass materials tensor as model input
            loss = criterion(output, thickness)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

        # Validation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for reflectance, thickness in val_loader:
                output = model(reflectance, materials_tensor)  # Pass materials tensor as model input
                val_loss += criterion(output, thickness).item()
        print(f'Validation Loss: {val_loss / len(val_loader)}')


# Train the model
train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=50)

# Save the trained model
torch.save(model.state_dict(), "transformer_model2.pth")


# Load the saved model
"""
model = TransformerModel(input_size, hidden_size, num_layers, num_heads, output_size, dropout=dropout)

model.load_state_dict(torch.load("transformer_model2.pth"))
model.eval()

# Define the loss function
criterion = nn.MSELoss()

# Convert test data to PyTorch tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader for the test set
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

test_loader2 = DataLoader(test_dataset, batch_size=10, shuffle=False)

# Evaluate the model on the test set
test_loss = 0.0
predictions = []
with torch.no_grad():
    for reflectance, thickness in test_loader2:
        output = model(reflectance, materials_tensor)  # Pass materials tensor as model input
        loss = criterion(output, thickness)
        test_loss += loss.item()
        predictions.append(output.numpy())

# Calculate test loss
avg_test_loss = test_loss / len(test_loader2)
print(f"Average Test Loss: {avg_test_loss}")

# Concatenate predictions and true values
predictions = np.concatenate(predictions, axis=0)
# Round predictions
rounded_predictions = np.round(predictions, decimals=2)
true_values = y_test.numpy()

# Print predicted and true values side by side
print("Predicted\t\tTrue")
for i in range(len(rounded_predictions)):
    print(f"{rounded_predictions[i]}\t\t{true_values[i]}")

# Calculate RMSE
rmse = mean_squared_error(true_values, rounded_predictions, squared=False)
print(f"Root Mean Squared Error (RMSE): {rmse}")
