#date: 2025-04-07T17:00:30Z
#url: https://api.github.com/gists/fb876a44edb0759f0757d31fb34c8125
#owner: https://api.github.com/users/SARAH-HADDAD

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy
import torch.nn.init as init
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class StrokeDataset(Dataset):
    def __init__(self, dataframe, feature_means=None, feature_stds=None):
        df = dataframe.copy()

        # Drop 'id' column
        df.drop(columns=["id"], inplace=True)

        # Handle missing values in 'bmi'
        df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
        df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

        # Encode categorical columns
        cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        for col in cat_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        self.labels = df['stroke'].values.astype('float32')
        self.features = df.drop(columns=['stroke']).values.astype('float32')

        if feature_means is None or feature_stds is None:
            self.feature_means = self.features.mean(axis=0)
            self.feature_stds = self.features.std(axis=0)
        else:
            self.feature_means = feature_means
            self.feature_stds = feature_stds

        # Normalize features
        self.features = (self.features - self.feature_means) / (self.feature_stds + 1e-8)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

    def get_normalization_params(self):
        # Return normalization parameters for test data
        return self.feature_means, self.feature_stds

# The model class:
class Net(nn.Module):
    def __init__(self, input_size=10, dropout_rate=0.2):
        super().__init__()
        # Define layers with appropriate sizes
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

        # Batch normalization helps stabilize training by normalizing layer inputs (Batch normalization addresses internal covariate shift and helps gradients flow)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(8)

        # Dropout for regularization (Dropout prevents overfitting by randomly deactivating neurons during training)
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize weights using He initialization for ReLU-family activations (He initialization is designed for ReLU/ELU activations and maintains variance)
        self._initialize_weights()

    def _initialize_weights(self):
        # He initialization for ELU layers
        init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu', a=1.0)
        init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu', a=1.0)
        # Different initialization for sigmoid layer
        init.xavier_normal_(self.fc3.weight)  # Xavier works better for sigmoid

    def forward(self, x):
        # First layer with ELU activation (ELU allows negative values and avoids the "dying ReLU" problem)
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.functional.elu(x)
        x = self.dropout(x)

        # Second layer with ELU activation
        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.functional.elu(x)
        x = self.dropout(x)

        # Output layer with sigmoid for binary classification
        x = torch.sigmoid(self.fc3(x))
        return x

def train_model(model, train_loader, test_loader, optimizer_name='adam',
                learning_rate=0.001, num_epochs=1000):
    # Select optimizer based on parameter (Different optimizers have different strengths for various tasks)
    if optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name.lower() == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99)
    else:  # Default to Adam
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # Binary Cross Entropy Loss for binary classification
    criterion = nn.BCELoss()

    # For tracking metrics
    train_losses = []
    test_accuracies = []

    print(f"Training with {optimizer_name} optimizer, learning rate: {learning_rate}")

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        for features, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels.view(-1, 1))

            # Backward pass and optimize
            loss.backward()
            # Gradient clipping helps prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        # Calculate epoch loss
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Evaluate on test set every 100 epochs
        if (epoch + 1) % 100 == 0 or epoch == num_epochs - 1:
            test_accuracy = evaluate_model(model, test_loader)
            test_accuracies.append(test_accuracy)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        else:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    return train_losses, test_accuracies

def evaluate_model(model, test_loader):
    accuracy_metric = Accuracy(task="binary")
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # No need to track gradients
        for features, labels in test_loader:
            outputs = model(features)
            preds = (outputs >= 0.5).float()
            accuracy_metric.update(preds, labels.view(-1, 1))

    return accuracy_metric.compute().item()

def plot_training_results(train_losses, test_accuracies, num_epochs):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot training loss
    ax1.plot(range(1, num_epochs+1), train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')

    # Plot test accuracy (only measured every 100 epochs)
    test_epochs = list(range(100, num_epochs+1, 100))
    if len(test_epochs) < len(test_accuracies):
        test_epochs.append(num_epochs)
    ax2.plot(test_epochs, test_accuracies)
    ax2.set_title('Test Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()



df = pd.read_csv("healthcare-dataset-stroke-data.csv")
# Train/Test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["stroke"])

# Create datasets
train_dataset = StrokeDataset(train_df)
feature_means, feature_stds = train_dataset.get_normalization_params()
test_dataset = StrokeDataset(test_df, feature_means, feature_stds)


# Create data loaders with proper batch size
# Appropriate batch size helps balance computation speed and gradient accuracy
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # Batch size 32 is a good starting point for most problems
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define model
input_size = train_dataset.features.shape[1]  # Get input size dynamically
model = Net(input_size=input_size)

# Train the model with Adam optimizer (Adam combines benefits of RMSprop and momentum for adaptive learning rates)
train_losses, test_accuracies = train_model(
        model,
        train_loader,
        test_loader,
        optimizer_name='adam',  # Try it also with : 'sgd', 'adagrad', 'rmsprop'
        learning_rate=0.001,
        num_epochs=1000 # 1000 epochs is excessive for real use - consider early stopping
        )

# Training results:
plot_training_results(train_losses, test_accuracies, 1000)

# Final evaluation
final_accuracy = evaluate_model(model, test_loader)
print(f"Final Test Accuracy: {final_accuracy:.4f}")

# Save the model
torch.save(model.state_dict(), "stroke_prediction_model.pth")
print("Model saved successfully")
