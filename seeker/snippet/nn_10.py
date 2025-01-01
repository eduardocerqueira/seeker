#date: 2025-01-01T16:54:25Z
#url: https://api.github.com/gists/72621af1ebeaf5aaea4c68bc1e42194c
#owner: https://api.github.com/users/PieroPaialungaAI

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Transformer-based model to transform sine to cosine
class SineToCosineTransformer(nn.Module):
    def __init__(self, num_layers=2, d_model=16, nhead=4, dim_feedforward=64, max_len=500):
        super(SineToCosineTransformer, self).__init__()
        self.positional_encoding = nn.Parameter(self.get_positional_encoding(max_len, d_model))
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=dim_feedforward)
        self.fc_in = nn.Linear(1, d_model)
        self.fc_out = nn.Linear(d_model, 1)
        self.relu = nn.ReLU()

    def get_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.fc_in(x)  # Map input to d_model dimension
        x += self.positional_encoding[:, :seq_len, :]  # Add positional encoding matching the input length
        x = self.transformer(x, x)
        x = self.fc_out(x)  # Map back to 1 dimension
        return x

# Generate sine wave data
def generate_sine_wave(frequency, length, sampling_rate):
    t = np.linspace(0, length, int(length * sampling_rate))
    sine_wave = np.sin(2 * np.pi * frequency * t)
    return torch.tensor(sine_wave, dtype=torch.float32).unsqueeze(1), torch.tensor(t, dtype=torch.float32)

# Main function
if __name__ == "__main__":
    # Parameters
    frequency = 1  # 1 Hz
    length = 2  # 2 seconds
    sampling_rate = 100  # 100 Hz

    # Generate sine wave and its corresponding cosine wave
    sine_wave, t = generate_sine_wave(frequency, length, sampling_rate)
    cosine_wave = torch.cos(2 * np.pi * frequency * t).unsqueeze(1)

    # Add batch dimension for Transformer
    sine_wave = sine_wave.unsqueeze(0)
    cosine_wave = cosine_wave.unsqueeze(0)

    # Initialize the transformer model
    model = SineToCosineTransformer(max_len=sine_wave.size(1))

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    epochs = 5000
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        output = model(sine_wave)
        loss = criterion(output, cosine_wave)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}')

    # Plot results
    with torch.no_grad():
        model.eval()
        predicted_cosine_wave = model(sine_wave).squeeze(0).detach().numpy()

    plt.figure(figsize=(10, 5))
    plt.plot(t.numpy(), sine_wave.squeeze(0).numpy(), label="Sine Wave")
    plt.plot(t.numpy(), predicted_cosine_wave, label="Predicted Cosine Wave", linestyle='--')
    plt.plot(t.numpy(), cosine_wave.squeeze(0).numpy(), label="Actual Cosine Wave", linestyle=':')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Sine to Cosine Transformation using Transformer')
    plt.show()
