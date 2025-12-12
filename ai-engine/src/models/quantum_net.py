import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNetLite(nn.Module):
    """
    QuantumNet-Lite: Optimized for M2 Pro (MPS) - 1-5m Framework
    Architecture: Conv1D -> LSTM -> Attention -> Heads
    Input: Sequence of candles (Batch, Seq_Len, Features)
    Output: 
        - Direction Probabilities (Wait, Buy, Sell)
        - Expected Return (Scalar)
    """
    def __init__(self, input_dim=33, hidden_dim=96, num_layers=2, num_heads=4, dropout=0.1):
        super(QuantumNetLite, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 1. Feature Extraction (Conv1D)
        # Input: (Batch, Features, Seq_Len) -> needs permute in forward
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=48, kernel_size=3, padding=1)
        self.conv_act = nn.ReLU()
        
        # 2. LSTM Encoder
        # Input to LSTM: (Batch, Seq_Len, 48)
        self.lstm = nn.LSTM(
            input_size=48,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False, # Plan implies single direction or unspecified? Usually Bi-LSTM is better, but let's stick to "LSTM(96)" which implies unidirect or standard. 
            # The user said "LSTM(96)". Often in simplified models this is unidirectional.
            # But let's check if Bi-LSTM was explicit. "Conv1D(48, kernel=3) → LSTM(96)".
            # If I make it bidirectional, output is 96*2. If unidirectional, 96. 
            # Let's assume unidirectional to match the "96" sizing exactly, or bidirectional=True and hidden=48?
            # Given "LSTM(96)", usually means hidden_size=96.
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 3. Multi-Head Attention
        # Input dimension is hidden_dim
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
        # 4. Normalization & Regularization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 5. Heads
        # Policy Head: 3 classes (Wait, Buy, Sell)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softmax(dim=-1)
        )
        
        # Value Head: Expected Return (Scalar)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Features)
        
        # Permute for Conv1D: (Batch, Features, Seq_Len)
        x = x.permute(0, 2, 1)
        
        # Conv1D
        x = self.conv1(x)
        x = self.conv_act(x)
        
        # Permute back for LSTM: (Batch, Seq_Len, 48)
        x = x.permute(0, 2, 1)
        
        # LSTM
        # out shape: (Batch, Seq_Len, Hidden)
        lstm_out, _ = self.lstm(x)
        
        # Attention
        # Self-attention over the sequence
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual Connection + Norm
        x = self.layer_norm(lstm_out + attn_out)
        
        # Pooling (Global Average Pooling across Sequence)
        # (Batch, Seq_Len, Hidden) -> (Batch, Hidden)
        x = torch.mean(x, dim=1)
        
        x = self.dropout(x)
        
        # Heads
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value

if __name__ == "__main__":
    # Test compatibility with MPS (Metal Performance Shaders)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Using Device: {device}")
    
    model = QuantumNetLite().to(device)
    model.eval() # Set to eval mode
    
    # Dummy Input: Batch=1, Seq=64, Feats=33
    dummy_input = torch.randn(1, 64, 33).to(device)
    
    # TorchScript Trace Test (Optimization for 1-5m latency)
    try:
        traced_script_module = torch.jit.trace(model, dummy_input)
        print("✅ TorchScript Trace Successful")
        policy, value = traced_script_module(dummy_input)
    except Exception as e:
        print(f"❌ TorchScript Trace Failed: {e}")
        policy, value = model(dummy_input)
    
    print("Policy Output:", policy)
    print("Value Output:", value)

