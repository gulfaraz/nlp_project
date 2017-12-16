from train import Train
import sys

train_easy = Train(difficulty="easy")
train_hard = Train(difficulty="hard")
learning_rates = [ 0.0001, 0.001, 0.01, 0.1, 1 ]
embedding_sizes = [ 200, 250, 300, 350, 400 ]
for lr in learning_rates:
    for es in embedding_sizes:
        train_easy(number_of_iterations=20, learning_rate=lr, embedding_size=es)
        train_hard(number_of_iterations=20, learning_rate=lr, embedding_size=es)
