import torch
import torch.nn as nn

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        
        # Initialize layers
        self.W = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_self = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_output = nn.Parameter(torch.Tensor(hidden_dim, output_dim))
        
        # Initialize Chebyshev coefficients
        self.num_coeffs = min(2,int(hidden_dim/2)) # truncated expansion order
        self.cheby_coeffs = nn.Parameter(torch.Tensor(self.num_coeffs, 1, hidden_dim))
        
        # Initialize biases
        self.bias_hidden = nn.Parameter(torch.Tensor(hidden_dim))
        self.bias_output = nn.Parameter(torch.Tensor(output_dim))
        
        # Initialize activation function
        #self.activation = nn.ReLU()
        
        # Initialize normalization matrix
        self.identity = nn.Parameter(torch.eye(input_dim))

        self.reset_parameters()        

    def reset_parameters(self):
        # Initialize parameters
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.W_self)
        nn.init.xavier_uniform_(self.W_output)
        nn.init.uniform_(self.cheby_coeffs, -1, 1)

        nn.init.normal_(self.bias_hidden, mean=0, std=0.1)
        nn.init.normal_(self.bias_output, mean=0, std=0.1)


    def forward(self, vertex_matrix, edge_matrix):
        # Extract vertex features
        V = vertex_matrix

        joined_columns = edge_matrix

        # Construct edge matrix
        E = torch.ones(joined_columns.size(0), joined_columns.size(0))
        E = torch.where(torch.eye(joined_columns.size(0)).bool(), E, E * 0.1)
        E += torch.eye(joined_columns.size(0))
        E /= E.sum(dim=0, keepdim=True)

        # Compute neighborhood matrix
        D = E.T @ V

        # Compute spectral filter
        cheby_inputs = [D]  # Initialize with neighborhood matrix D

        for i in range(1, self.num_coeffs):
            if len(cheby_inputs) < 2:
                cheby_inputs.append(2 * (E @ cheby_inputs[-1]))
            else:
                cheby_inputs.append(2 * (E @ cheby_inputs[-1]) - cheby_inputs[-2])

        cheby_coeffs = self.cheby_coeffs.view(self.num_coeffs, -1)
        cheby_outputs = torch.zeros_like(D)
        for coeff, input in zip(cheby_coeffs, cheby_inputs):
            coeff_expanded = coeff.unsqueeze(0).repeat(input.size(0), 1)
            cheby_outputs += coeff_expanded[:, :input.size(1)] * input

        bias_hidden_expanded = self.bias_hidden[:cheby_outputs.size(1)].unsqueeze(0)
        #V = self.activation(cheby_outputs + bias_hidden_expanded)
        V = cheby_outputs + bias_hidden_expanded

        # Compute output
        output = V @ self.W_output[:V.size(-1), :] + self.bias_output

        return output, V
