import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelBinarizer
import time

class MNISTNeuralNetwork:
    def __init__(self, hidden_size=128):
        """
        MNIST Neural Network Architecture:
        784 inputs (28x28 pixels) → hidden_size neurons → 10 outputs (digits 0-9)
        
        This is like your XOR network but MUCH bigger!
        """
        self.input_size = 784    # 28 × 28 = 784 pixels
        self.hidden_size = hidden_size  # You can experiment with this!
        self.output_size = 10    # 10 digits (0, 1, 2, ..., 9)
        
        print(f"🧠 Creating MNIST Neural Network:")
        print(f"   📥 Input layer: {self.input_size} neurons (28×28 pixels)")
        print(f"   🔄 Hidden layer: {self.hidden_size} neurons")  
        print(f"   📤 Output layer: {self.output_size} neurons (digits 0-9)")
        
        # Initialize weights (same concept as XOR!)
        np.random.seed(42)
        
        # Weights: Input → Hidden (784 × hidden_size)
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.1
        self.b1 = np.zeros((1, self.hidden_size))
        
        # Weights: Hidden → Output (hidden_size × 10)  
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.1
        self.b2 = np.zeros((1, self.output_size))
        
        print(f"   ⚡ Weights initialized!")
        print(f"   W1 shape: {self.W1.shape}, W2 shape: {self.W2.shape}")
    
    def sigmoid(self, x):
        """Same sigmoid function from XOR network"""
        x = np.clip(x, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Same derivative from XOR network"""
        return x * (1 - x)
    
    def softmax(self, x):
        """
        Softmax: Better than sigmoid for multiple classes
        Converts raw scores to probabilities that sum to 1
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        Forward pass - same concept as XOR but with softmax output
        """
        # Input → Hidden (with sigmoid activation)
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Hidden → Output (with softmax activation for probability)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)  # Probabilities for each digit
        
        return self.a2
    
    def backward(self, X, y, learning_rate=0.1):
        """
        Backpropagation - same concept as XOR but for 10 classes
        """
        m = X.shape[0]  # Number of examples
        
        # Calculate loss (cross-entropy for classification)
        loss = -np.mean(np.sum(y * np.log(self.a2 + 1e-8), axis=1))
        
        # Backward pass (same logic as XOR!)
        # Output layer error
        output_error = self.a2 - y
        output_delta = output_error
        
        # Hidden layer error  
        hidden_error = output_delta.dot(self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a1)
        
        # Update weights (same as XOR!)
        self.W2 -= learning_rate * self.a1.T.dot(output_delta) / m
        self.b2 -= learning_rate * np.sum(output_delta, axis=0, keepdims=True) / m
        
        self.W1 -= learning_rate * X.T.dot(hidden_delta) / m  
        self.b1 -= learning_rate * np.sum(hidden_delta, axis=0, keepdims=True) / m
        
        return loss
    
    def predict(self, X):
        """Make predictions - return the digit with highest probability"""
        predictions = self.forward(X)
        return np.argmax(predictions, axis=1)
    
    def train(self, X, y, epochs=100, learning_rate=0.1, batch_size=100):
        """
        Train on MNIST data with mini-batches for efficiency
        """
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        
        print(f"\n🚀 Training on {n_samples:,} images for {epochs} epochs...")
        print(f"📦 Using mini-batches of size {batch_size}")
        print("\nEpoch | Loss    | Accuracy | Time")
        print("------|---------|----------|-----")
        
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            start_time = time.time()
            epoch_loss = 0
            
            # Shuffle data each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch training
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward and backward pass
                self.forward(X_batch)
                loss = self.backward(X_batch, y_batch, learning_rate)
                epoch_loss += loss
            
            # Calculate average loss and accuracy
            avg_loss = epoch_loss / n_batches
            predictions = self.predict(X)
            accuracy = np.mean(predictions == np.argmax(y, axis=1)) * 100
            
            losses.append(avg_loss)
            accuracies.append(accuracy)
            
            # Print progress
            elapsed = time.time() - start_time
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"{epoch:5d} | {avg_loss:.5f} | {accuracy:7.2f}% | {elapsed:.1f}s")
        
        return losses, accuracies

def load_and_prepare_mnist():
    """Load MNIST data and prepare it for our network"""
    print("📥 Loading MNIST dataset...")
    
    # Load MNIST data (this might take a moment first time)
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target
    
    # Convert to proper types
    X = X.astype('float32')
    y = y.astype('int64')
    
    print(f"✅ Loaded {X.shape[0]:,} images of size {int(np.sqrt(X.shape[1]))}×{int(np.sqrt(X.shape[1]))}")
    
    # Normalize pixel values to 0-1 range (helps training)
    X = X / 255.0
    
    # Convert labels to one-hot encoding (like [0,0,1,0,0,0,0,0,0,0] for digit 2)
    label_binarizer = LabelBinarizer()
    y_one_hot = label_binarizer.fit_transform(y)
    
    # Use smaller subset for faster training (you can increase this later)
    n_samples = 10000  # Start small, increase later!
    X_subset = X[:n_samples]
    y_subset = y_one_hot[:n_samples]
    
    print(f"🎯 Using {n_samples:,} samples for training")
    print(f"📊 Input shape: {X_subset.shape}")
    print(f"📊 Output shape: {y_subset.shape}")
    
    return X_subset, y_subset, X, y

def visualize_predictions(network, X, y_true, n_samples=10):
    """Show some predictions to see how well the network is doing"""
    # Get random samples
    indices = np.random.choice(len(X), n_samples, replace=False)
    
    # Make predictions
    predictions = network.predict(X[indices])
    probabilities = network.forward(X[indices])
    
    # Plot results
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(n_samples):
        # Reshape image back to 28x28
        image = X[indices[i]].reshape(28, 28)
        
        # Plot image
        axes[i].imshow(image, cmap='gray')
        
        # Get true label and prediction
        true_label = y_true[indices[i]]
        pred_label = predictions[i]
        confidence = probabilities[i][pred_label] * 100
        
        # Color: green if correct, red if wrong
        color = 'green' if pred_label == true_label else 'red'
        
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)', 
                         color=color, fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("🎯 MNIST Digit Recognition from Scratch!")
    print("=" * 50)
    
    # Load data
    X_train, y_train, X_full, y_full = load_and_prepare_mnist()
    
    # Create and train network
    network = MNISTNeuralNetwork(hidden_size=128)
    
    # Train the network
    losses, accuracies = network.train(X_train, y_train, 
                                     epochs=50, 
                                     learning_rate=0.5,
                                     batch_size=100)
    
    # Final evaluation
    print(f"\n🎉 Training Complete!")
    final_accuracy = accuracies[-1]
    print(f"📊 Final Accuracy: {final_accuracy:.2f}%")
    
    if final_accuracy > 85:
        print("🏆 Excellent! You beat 85% accuracy!")
    elif final_accuracy > 70:
        print("👍 Good job! Getting close to expert level!")
    else:
        print("📈 Keep training! Try more epochs or adjust learning rate.")
    
    # Plot training progress
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Show some predictions as examples
    print("\n🔍 Let's see some predictions...")
    visualize_predictions(network, X_train, np.argmax(y_train, axis=1))
    
    print("\n" + "=" * 50)
    print("🚀 Next steps:")
    print("1. Try increasing epochs to 100+")
    print("2. Experiment with hidden_size (64, 256, 512)")
    print("3. Try different learning rates (0.1, 1.0)")
    print("4. Use the full 60,000 samples!")
    print("=" * 50)