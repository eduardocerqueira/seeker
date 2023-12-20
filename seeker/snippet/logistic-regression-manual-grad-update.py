#date: 2023-12-20T16:50:31Z
#url: https://api.github.com/gists/18711a3cc6f7ce0e476456d2a67e14de
#owner: https://api.github.com/users/FaizalKarim280280

import torch
from sklearn.datasets import make_regression, make_classification
from sklearn.preprocessing import StandardScaler

class Trainer:
    def __init__(self, num_features):
        self.lr = 5e-2
        self. w = torch.randn(num_features + 1, 1, requires_grad = True)

    def train(self, X, y, epochs = 1000):
        for i in range(epochs):
            y_pred = (X @ self.w).view(-1)
            y_pred = torch.nn.Sigmoid()(y_pred)
            loss = -torch.mean(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred))
            acc = torch.sum(((y_pred >= 0.5) == y))/len(y)
            loss.backward()

            with torch.no_grad():
                self.w -= self.lr * self.w.grad
                self.w.grad.zero_()

            if (i + 1) % 100 == 0:
                print(f"Loss:{loss.item():.4f} Acc:{acc.item():.4f}")
                
def main():
    num_features = 6
    X, y = make_classification(1000, num_features, n_classes=2, n_redundant = 3)
    X_scaler = StandardScaler()

    X = X_scaler.fit_transform(X)

    X = np.hstack((np.ones(len(X)).reshape(-1, 1), X))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)
    
    trainer = Trainer(num_features)
    trainer.train(epochs = 5000)
    
if __name__ == "__main__":
    main()