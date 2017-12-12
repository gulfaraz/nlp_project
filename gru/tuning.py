from train import Train
import sys

difficulty = "easy"

if len(sys.argv) > 1:
    difficulty = sys.argv[1]

if difficulty != "hard":
    difficulty = "easy"

train = Train(difficulty=difficulty)
learning_rates = [ 0.001, 0.01 ]
embedding_sizes = [ 300, 30 ]
for lr in learning_rates:
    for es in embedding_sizes:
        train(number_of_iterations=1, learning_rate=lr, embedding_size=es)
