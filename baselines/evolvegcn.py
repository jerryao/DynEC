import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EvolveGCN_O_Layer(nn.Module):
    """
    EvolveGCN-O Layer: Evolves the GCN weight matrix using an LSTM.
    The weight matrix W_t is treated as the hidden state of the LSTM.
    """
    def __init__(self, in_channels, out_channels):
        super(EvolveGCN_O_Layer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # LSTM to evolve weights
        # We treat the flattened weight matrix as the hidden state.
        # Input dimension: We can use a static input or the previous weight.
        # Paper suggests using node features (EvolveGCN-H) or nothing (EvolveGCN-O).
        # For EvolveGCN-O, we use a learnable static parameter as input.
        self.lstm_input_dim = in_channels * out_channels
        self.lstm = nn.LSTMCell(self.lstm_input_dim, self.lstm_input_dim)
        
        # Initial weight and cell state
        # W_0 is the initial hidden state h_0
        self.W_init = nn.Parameter(torch.Tensor(in_channels, out_channels))
        # Initial cell state c_0
        self.C_init = nn.Parameter(torch.Tensor(in_channels * out_channels))
        
        # Static input for the LSTM (since no node features are used in O-version)
        self.static_input = nn.Parameter(torch.Tensor(self.lstm_input_dim))
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W_init.size(1))
        self.W_init.data.uniform_(-stdv, stdv)
        self.C_init.data.zero_()
        self.static_input.data.zero_()

    def forward(self, hx=None):
        """
        Evolve one step.
        hx: (h, c) tuple from previous step. If None, use initial parameters.
        Returns: W_t (reshaped h), (h_next, c_next)
        """
        if hx is None:
            h = self.W_init.reshape(1, -1) # (1, in*out)
            c = self.C_init.reshape(1, -1) # (1, in*out)
        else:
            h, c = hx
            
        # LSTM Step
        # Input is static parameter (expanded to batch size 1)
        input_obs = self.static_input.reshape(1, -1)
        
        h_next, c_next = self.lstm(input_obs, (h, c))
        
        # Reshape hidden state to be the Weight Matrix
        W_t = h_next.reshape(self.in_channels, self.out_channels)
        
        return W_t, (h_next, c_next)

class EvolveGCN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, n_layers=2):
        super(EvolveGCN, self).__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        
        # We can have multiple layers. For simplicity, let's do 2 layers:
        # Layer 1: in -> hidden
        self.layers.append(EvolveGCN_O_Layer(in_feat, hidden_feat))
        
        # Layer 2: hidden -> out (if n_layers > 1)
        # Note: EvolveGCN usually has GCN layers evolved by RNN.
        if n_layers > 1:
            self.layers.append(EvolveGCN_O_Layer(hidden_feat, out_feat))
        
        self.act = nn.ReLU()

    def forward(self, x_seq, adj_seq, hx_list=None):
        """
        x_seq: (T, N, F) - Node features sequence
        adj_seq: List of T adjacency matrices (N, N) (Sparse or Dense Tensor)
        hx_list: List of hidden states for each layer. If None, initialized to None.
        """
        T = x_seq.size(0)
        outputs = []
        
        # Initialize LSTM states for each layer if not provided
        if hx_list is None:
            hx_list = [None] * len(self.layers)
        
        for t in range(T):
            x = x_seq[t] # (N, F)
            adj = adj_seq[t] # (N, N)
            
            # Forward through layers
            for i, layer in enumerate(self.layers):
                # 1. Evolve Weights
                W_t, hx_next = layer(hx_list[i])
                hx_list[i] = hx_next
                
                # 2. GCN Operation: H = A * X * W
                # X * W
                support = torch.mm(x, W_t)
                
                # A * support
                if adj.is_sparse:
                    x = torch.spmm(adj, support)
                else:
                    x = torch.mm(adj, support)
                
                # Activation (except last layer? Usually last layer also has act or just logits)
                # We'll apply ReLU for intermediate layers
                if i < len(self.layers) - 1:
                    x = self.act(x)
            
            # x is now the embedding for time t
            outputs.append(x)
            
        return torch.stack(outputs), hx_list
