#date: 2023-12-22T16:48:59Z
#url: https://api.github.com/gists/62a0cd2169fe3ca5de923213b4e089a5
#owner: https://api.github.com/users/d0rc

import torch
from torch import nn
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

device = 'cpu'

# Custom Dataset
# Function to read data from CSV and convert to tensor
def read_data(file_path, device):
    df = pd.read_csv(file_path, nrows=100_000)

    # Convert strings to lists of floats and then to tensors
    inputs = df['input'].apply(lambda x: torch.tensor([float(i.replace("'", "")) for i in x.split(';')], dtype=torch.float))
    outputs = df['output'].apply(lambda x: torch.tensor([float(i.replace("'", "")) for i in x.split(';')], dtype=torch.float))

    # Normalize function that uses min and max from the input vector
    def normalize_vector(input_vector, output_vector):
        #min_val = input_vector.min()
        #max_val = input_vector.max()
        #normalized_input = (input_vector - min_val) / (max_val - min_val)
        #normalized_output = (output_vector - min_val) / (max_val - min_val)
        #return normalized_input, normalized_output
        return input_vector, output_vector

    # Apply normalization to each pair of vectors
    normalized_data = [normalize_vector(input_vector, output_vector) for input_vector, output_vector in zip(inputs, outputs)]
    inputs_normalized, outputs_normalized = zip(*normalized_data)

    # Convert to tensors
    inputs_normalized = torch.stack(inputs_normalized)
    outputs_normalized = torch.stack(outputs_normalized)

    return inputs_normalized.to(device), outputs_normalized.to(device)



# Custom Dataset class
class VectorPairDataset(Dataset):
    def __init__(self, file_path):
        self.inputs, self.targets = read_data(file_path, device)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_hidden_layers):
        super(SimpleNN, self).__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.n_hidden_layers = n_hidden_layers
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.innerLayers = nn.ModuleList()
        [self.innerLayers.append(nn.Linear(hidden_dim, hidden_dim, bias=True)) for _ in range(self.n_hidden_layers)]
        self.nonLinearity = nn.LeakyReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x):
        x = self.linear1(self.norm(x))
        x = self.nonLinearity(x)

        for idx in range(self.n_hidden_layers):
            x = self.innerLayers[idx](x)
            x = self.nonLinearity(x)

        x = self.linear2(x)
        return x

def old_main():
    # Parameters
    input_dim = 4096
    hidden_dim = 8192*2
    depth = 4
    output_dim = 4096
    batch_size = 64
    learning_rate = 0.0001
    num_epochs = 10000

    # Dataset and DataLoader
    dataset = VectorPairDataset("/tmp/paired_embeddings.csv")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, Loss and Optimizer
    model = SimpleNN(input_dim, hidden_dim, output_dim, depth).to(device)
    # model = torch.compile(model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        # Set up the progress bar
        # The total parameter is set to the length of the dataloader to show the progress over the entire dataset
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1}/{num_epochs}')

        total_loss = 0  # To accumulate loss over the epoch
        for i, (inputs, targets) in progress_bar:
            # Forward pass
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.to(device))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update progress bar
            # You can add more metrics as needed
            progress_bar.set_postfix(loss=loss.item(), total_loss=total_loss / (i + 1))

        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {total_loss / len(dataloader):.4f}')

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, input_dim, hid_dim, n_heads, dropout_rate):
        super().__init__()

        self.wq = nn.Linear(input_dim, hid_dim, bias=True)
        self.wk = nn.Linear(input_dim, hid_dim, bias=True)
        self.wv = nn.Linear(input_dim, hid_dim, bias=True)
        self.multi_head_attention = nn.MultiheadAttention(hid_dim, n_heads, dropout=dropout_rate)
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hid_dim, input_dim, bias=True)

    def forward(self, x):
        # Multi-head attention
        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)
        attention_output, _ = self.multi_head_attention(query, key, value)

        # Add & Norm
        x = self.layer_norm(query + self.dropout(attention_output))

        return self.fc(x)

class VectorFormer(nn.Module):
    def __init__(self, n_layers, input_dim, hid_dim, n_heads, dropout_rate):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(MultiHeadAttentionLayer(hid_dim=hid_dim,
                                                       input_dim=input_dim,
                                                       n_heads=n_heads,
                                                       dropout_rate=dropout_rate))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

def x_main():
    # Parameters
    # Example usage # You can tune this parameter
    batch_size = 1
    learning_rate = 0.001
    num_epochs = 10000

    #model = GRUModel(4096, 4096, 4096).to(device)
    #model = torch.compile(model)
    model = VectorFormer(16, 4096,128, 32, 0.01).to(device)

    # Dataset and DataLoader
    dataset = VectorPairDataset("/tmp/paired_embeddings.csv")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, Loss and Optimizer
    # model = SimpleNN(input_dim, hidden_dim, output_dim, depth).to(device)
    # model = torch.compile(model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        # Set up the progress bar
        # The total parameter is set to the length of the dataloader to show the progress over the entire dataset
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1}/{num_epochs}')

        total_loss = 0  # To accumulate loss over the epoch
        for i, (targets, inputs) in progress_bar:
            # Forward pass
            inputs = inputs.to(device)
            outputs = model(inputs)
            # loss = criterion(outputs, targets.to(device))
            loss = torch.norm(outputs - targets.to(device))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update progress bar
            # You can add more metrics as needed
            progress_bar.set_postfix(loss=loss.item(), total_loss=total_loss / (i + 1))

        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {total_loss / len(dataloader):.4f}')

# Execute the program
if __name__ == "__main__":
    x_main()
