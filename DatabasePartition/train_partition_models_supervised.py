import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from api.services.partition.database import database
from api.services.partition.partition_evaluation.evaluation_model import SampleGraph, partition_evaluation_model
from api.services.partition.config import PartitionConfig
import numpy as np
from sklearn.preprocessing import StandardScaler
import time

def load_and_preprocess_data(data):
    
    # Assuming data is a list of tuples (features, label) where features is a numpy array and label is a float
    graph_features = [sample[0] for sample in data]
    label_list = [list(sample[1].values()) for sample in data]  # Convert dict values to list
    label_list = np.array(label_list)

    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    normalized_labels = scaler.fit_transform(label_list)

    normalized_data = []
    for i, graph_feature in enumerate(graph_features):
        normalized_data.append((graph_feature, normalized_labels[i]))

    return normalized_data, scaler


def normalize_new_label(scaler, new_label_value):

    new_label_value = new_label_value.data


    # Use the scaler to transform the new label value(s)
    # array([3.73213700e+03, 2.67943004e-04])
    normalized_new_label = scaler.transform(np.array([[new_label_value[0], new_label_value[0]]]))
    new_label_value.data[0] = float(normalized_new_label[0][0])

    return new_label_value


def exp_training():
    # Load the serialized data (274)
    with open('distinct_sampled_training_data.pickle', 'rb') as f:
        data = pickle.load(f)
    
    data, scaler = load_and_preprocess_data(data)
    # Splitting the data into training and testing
    split_index = int(len(data) * 0.8)
    training_samples, testing_samples = data[:split_index], data[split_index:]

    num_epochs = 50
    args = PartitionConfig()
    success, msg = args.generate_paths()
    if not success:
        raise ValueError(msg)

    if os.path.exists("./logs/self_supervised_training_log.txt"):
        os.remove("./logs/self_supervised_training_log.txt")
    with open("./logs/self_supervised_training_log.txt", "w") as f:
        f.write("")
    
    if not os.path.exists(args.pretrain_model_checkpoint):
        os.makedirs(args.pretrain_model_checkpoint)

    # args.max_node_num (10-20)
    # args.evaluation_learning_rate (0.001 to 0.1)
    for max_node_num_value in range(10, 21):
        args.max_node_num = max_node_num_value
        for learning_rate_value in np.linspace(0.001, 0.1, 10):
            # Training evaluation model with max_node_num: 15 and learning_rate: 0.05600000000000001
            # Test Loss: 0.6954802525483749

            args.evaluation_learning_rate = learning_rate_value

            e_model = partition_evaluation_model(args)
            e_optimizer = optim.Adam(list(e_model.gnn.parameters()) + list(e_model.fc_layer.parameters()), lr=args.evaluation_learning_rate)
            e_criterion = nn.MSELoss()

            for epoch in range(num_epochs):
                e_model.gnn.train()  # Ensure the model is in training mode
                for sample in training_samples:
                    partitioned_sample_graph = sample[0]
                    embedding = e_model.embedding(partitioned_sample_graph)
                    estimated_latency = e_model.estimate_latency(embedding)
                    estimated_latency = normalize_new_label(scaler, estimated_latency)

                    labels = sample[1]
                    real_latency = torch.tensor(labels[0], dtype=torch.float, requires_grad=True)

                    loss = e_criterion(real_latency, estimated_latency)
                    e_optimizer.zero_grad()
                    loss.backward()
                    e_optimizer.step()

            e_model.gnn.eval()  # Set the model to evaluation mode for testing
            test_loss = 0
            with torch.no_grad():  # Disable gradient computation during evaluation
                for sample in testing_samples:
                    partitioned_sample_graph = sample[0]
                    embedding = e_model.embedding(partitioned_sample_graph)
                    estimated_latency = e_model.estimate_latency(embedding)
                    estimated_latency = normalize_new_label(scaler, estimated_latency)

                    labels = sample[1]
                    real_latency = torch.tensor(labels[0], dtype=torch.float)
                    
                    test_loss += e_criterion(real_latency, estimated_latency).item()

            with open("./logs/self_supervised_training_log.txt", "a") as f:
                f.write(f"Training evaluation model with max_node_num: {max_node_num_value} and learning_rate: {learning_rate_value}\n")
                f.write(f"Test Loss: {test_loss / len(testing_samples)}\n")
            print(f"Training evaluation model with max_node_num: {max_node_num_value} and learning_rate: {learning_rate_value}")            
            print(f"Test Loss: {test_loss / len(testing_samples)}")

            if test_loss / len(testing_samples) < 0.1:
                print("Test Passed")
                torch.save(e_model.gnn.state_dict(), os.path.join(args.pretrain_model_checkpoint, f'evaluation_model_{max_node_num_value}_{learning_rate_value}.pth'))
                exit(1)
            else:
                print("Test Failed")

            time.sleep(2)

if __name__ == "__main__":
    exp_training()