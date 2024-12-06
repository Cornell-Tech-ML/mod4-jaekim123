import random
import minitorch


# Linear Layer Class
class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = []
        self.bias = []
        # Initialize the weights and biases randomly
        for i in range(in_size):
            self.weights.append([])
            for j in range(out_size):
                self.weights[i].append(
                    self.add_parameter(
                        f"weight_{i}_{j}", minitorch.Scalar(2 * (random.random() - 0.5))
                    )
                )
        for j in range(out_size):
            self.bias.append(
                self.add_parameter(
                    f"bias_{j}", minitorch.Scalar(2 * (random.random() - 0.5))
                )
            )

    def forward(self, inputs):
        # Perform the linear transformation Wx + b using Scalar objects
        output = []
        for j in range(len(self.bias)):
            result = self.bias[j].value  # Access the underlying value of the bias
            for i in range(len(inputs)):
                result += (
                    inputs[i] * self.weights[i][j].value
                )  # Use .value to get the Scalar value
            output.append(
                result
            )  # Directly append the result, no need to wrap it in Scalar
        return output  # Return the list of results


# Neural Network Class
class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        # Define the layers using Linear transformations
        self.layer1 = Linear(2, hidden_layers)  # 2 input features -> hidden_layers
        self.layer2 = Linear(hidden_layers, hidden_layers)  # hidden -> hidden
        self.layer3 = Linear(hidden_layers, 1)  # hidden -> 1 output feature

    def forward(self, x):
        # Forward pass through the network
        middle = [h.relu() for h in self.layer1.forward(x)]
        middle2 = [h.relu() for h in self.layer2.forward(middle)]
        return self.layer3.forward(middle2)[0].sigmoid()  # Sigmoid for final output


# Default logging function
def default_log_fn(epoch, total_loss, correct, losses):
    print(f"Epoch {epoch}, Loss: {total_loss}, Correct: {correct}/{len(losses)}")


# Training Class
class ScalarTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(self.hidden_layers)

    def run_one(self, x):
        return self.model.forward(
            (minitorch.Scalar(x[0], name="x_1"), minitorch.Scalar(x[1], name="x_2"))
        )

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward pass and loss computation
            for i in range(data.N):
                x_1, x_2 = data.X[i]
                y = data.y[i]
                x_1 = minitorch.Scalar(x_1)
                x_2 = minitorch.Scalar(x_2)
                out = self.model.forward((x_1, x_2))

                if y == 1:
                    prob = out
                    correct += 1 if out.data > 0.5 else 0
                else:
                    prob = -out + 1.0
                    correct += 1 if out.data < 0.5 else 0

                # Negative log likelihood loss
                loss = -prob.log()
                (loss / data.N).backward()  # Normalized by the number of data points
                total_loss += loss.data

            losses.append(total_loss)

            # Update weights
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                log_fn(epoch, total_loss, correct, losses)


# Main function to run the training
if __name__ == "__main__":
    # Training configuration
    PTS = 50  # Number of data points
    HIDDEN = 10  # Number of hidden units in each layer
    RATE = 0.1  # Learning rate

    # Load dataset
    data = minitorch.datasets["Simple"](PTS)  # Simple dataset for training

    # Train the model
    ScalarTrain(HIDDEN).train(data, RATE)
