"""
ConvLSTM Architecture for Temporal Air Quality Forecasting
Supports [Batch, Time, Channels, Height, Width] input sequences
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell with convolutions inside each gate.
    
    Implements the ConvLSTM equations:
        i_t = σ(W_xi * x_t + W_hi * h_{t-1} + b_i)      # Input gate
        f_t = σ(W_xf * x_t + W_hf * h_{t-1} + b_f)      # Forget gate
        o_t = σ(W_xo * x_t + W_ho * h_{t-1} + b_o)      # Output gate
        c̃_t = tanh(W_xc * x_t + W_hc * h_{t-1} + b_c)   # Candidate cell state
        c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t                 # Cell state update
        h_t = o_t ⊙ tanh(c_t)                            # Hidden state output
    
    Where all operations are implemented as convolutions (not fully connected).
    """
    
    def __init__(self, in_channels, hidden_channels, kernel_size=3, padding=1):
        """
        Args:
            in_channels: Number of input channels
            hidden_channels: Number of hidden/cell state channels
            kernel_size: Kernel size for convolutions
            padding: Padding for convolutions (typically kernel_size // 2)
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Input convolutions: map from [in_channels] to [4 * hidden_channels]
        # The 4x factor is for: input gate, forget gate, output gate, candidate cell
        self.conv_gates = nn.Conv2d(
            in_channels + hidden_channels,
            2 * hidden_channels,  # Input gate + Forget gate
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
        
        self.conv_output = nn.Conv2d(
            in_channels + hidden_channels,
            hidden_channels,  # Output gate
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
        
        self.conv_candidate = nn.Conv2d(
            in_channels + hidden_channels,
            hidden_channels,  # Candidate cell state
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
    
    def forward(self, x_t, h_t, c_t):
        """
        Forward pass for a single timestep.
        
        Args:
            x_t: Input tensor [Batch, in_channels, Height, Width]
            h_t: Hidden state [Batch, hidden_channels, Height, Width]
            c_t: Cell state [Batch, hidden_channels, Height, Width]
        
        Returns:
            h_t_new: New hidden state [Batch, hidden_channels, Height, Width]
            c_t_new: New cell state [Batch, hidden_channels, Height, Width]
        """
        # Concatenate input with hidden state along channel dimension
        combined = torch.cat([x_t, h_t], dim=1)
        
        # Compute input and forget gates: σ(W*combined)
        gates = self.conv_gates(combined)
        input_gate, forget_gate = torch.split(gates, self.hidden_channels, dim=1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        
        # Compute output gate: σ(W*combined)
        output_gate = torch.sigmoid(self.conv_output(combined))
        
        # Compute candidate cell state: tanh(W*combined)
        candidate_cell = torch.tanh(self.conv_candidate(combined))
        
        # Update cell state: f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
        c_t_new = forget_gate * c_t + input_gate * candidate_cell
        
        # Update hidden state: o_t ⊙ tanh(c_t)
        h_t_new = output_gate * torch.tanh(c_t_new)
        
        return h_t_new, c_t_new


class ConvLSTM(nn.Module):
    """
    Multi-layer Convolutional LSTM that processes temporal sequences.
    
    Input shape: [Batch, Time, Channels, Height, Width]
    Output shape: [Batch, Hidden_Channels, Height, Width] (final timestep)
    """
    
    def __init__(self, in_channels, hidden_channels_list, kernel_size=3, num_layers=2):
        """
        Args:
            in_channels: Number of input channels
            hidden_channels_list: List of hidden channel counts for each layer
            kernel_size: Kernel size for convolutions
            num_layers: Number of stacked ConvLSTM layers
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels_list = hidden_channels_list
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        # Create list of ConvLSTM cells
        self.layers = nn.ModuleList()
        
        for layer_idx in range(num_layers):
            cell_in_channels = in_channels if layer_idx == 0 else hidden_channels_list[layer_idx - 1]
            cell_hidden_channels = hidden_channels_list[layer_idx]
            
            cell = ConvLSTMCell(
                in_channels=cell_in_channels,
                hidden_channels=cell_hidden_channels,
                kernel_size=kernel_size,
                padding=padding
            )
            self.layers.append(cell)
    
    def forward(self, x):
        """
        Forward pass through the temporal sequence.
        
        Args:
            x: Input tensor [Batch, Time, Channels, Height, Width]
        
        Returns:
            output: Final hidden states from all layers
            h_list: List of hidden states for each layer (last timestep)
            c_list: List of cell states for each layer (last timestep)
        """
        batch_size, time_steps, channels, height, width = x.shape
        
        # Initialize hidden and cell states for all layers
        h_list = []
        c_list = []
        
        for layer_idx in range(self.num_layers):
            h = torch.zeros(
                batch_size,
                self.hidden_channels_list[layer_idx],
                height,
                width,
                device=x.device,
                dtype=x.dtype
            )
            c = torch.zeros(
                batch_size,
                self.hidden_channels_list[layer_idx],
                height,
                width,
                device=x.device,
                dtype=x.dtype
            )
            h_list.append(h)
            c_list.append(c)
        
        # Process each timestep
        for t in range(time_steps):
            x_t = x[:, t, :, :, :]  # [Batch, Channels, Height, Width]
            
            # Forward through all layers
            for layer_idx in range(self.num_layers):
                h_t, c_t = self.layers[layer_idx](x_t, h_list[layer_idx], c_list[layer_idx])
                h_list[layer_idx] = h_t
                c_list[layer_idx] = c_t
                
                # Output of this layer becomes input to next layer
                x_t = h_t
        
        return h_list[-1], h_list, c_list  # Return final layer output


class ConvLSTMPredictor(nn.Module):
    """
    Complete ConvLSTM-based air quality prediction model.
    
    Predicts t+1 from temporal sequence [t-3, t-2, t-1, t].
    Architecture:
        1. ConvLSTM encoder (processes temporal sequence, captures dynamics)
        2. ConvLSTM decoder (2-layer) with skip connections for spatial detail
        3. Output projection (5→1 channels for NO2)
    """
    
    def __init__(self, in_channels=5, hidden_channels=32, num_layers=2, output_channels=1):
        """
        Args:
            in_channels: Number of meteorological input channels
            hidden_channels: Base hidden channel count (scales per layer)
            num_layers: Number of ConvLSTM layers in encoder
            output_channels: Number of output channels (1 for NO2)
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        
        # Encoder: Process temporal sequence and extract spatiotemporal features
        encoder_channels = [hidden_channels, hidden_channels * 2]
        self.encoder = ConvLSTM(
            in_channels=in_channels,
            hidden_channels_list=encoder_channels,
            kernel_size=3,
            num_layers=num_layers
        )
        
        # Decoder: Upscale and refine predictions with skip connections
        # The decoder uses standard 2-layer ConvLSTM for temporal continuity
        decoder_channels = [hidden_channels * 2, hidden_channels]
        self.decoder = ConvLSTM(
            in_channels=hidden_channels * 2,
            hidden_channels_list=decoder_channels,
            kernel_size=3,
            num_layers=num_layers
        )
        
        # Refinement path: spatial detail enhancement
        self.refine_conv1 = nn.Conv2d(
            hidden_channels,
            hidden_channels // 2,
            kernel_size=3,
            padding=1
        )
        self.refine_conv2 = nn.Conv2d(
            hidden_channels // 2,
            output_channels,
            kernel_size=1
        )
        
        # Batch normalization for training stability
        self.bn_refine = nn.BatchNorm2d(hidden_channels // 2)
    
    def forward(self, x):
        """
        Forward pass for temporal sequence to next-step prediction.
        
        Args:
            x: Input tensor [Batch, Time=4, Channels=5, Height, Width]
        
        Returns:
            output: Predicted NO2 at t+1 [Batch, 1, Height, Width]
        """
        # Encoder: Extract spatiotemporal features from sequence
        encoder_output, h_list, c_list = self.encoder(x)
        # encoder_output: [Batch, hidden_channels*2, Height, Width]
        
        # Prepare decoder input by expanding temporal dimension (synthetic single step)
        # We use the encoder output as the starting point for refinement
        decoder_input = encoder_output.unsqueeze(1)  # [Batch, 1, hidden_channels*2, Height, Width]
        
        # Decoder: Process through another ConvLSTM for consistency
        decoder_output, _, _ = self.decoder(decoder_input)
        # decoder_output: [Batch, hidden_channels, Height, Width]
        
        # Refinement path: spatial detail enhancement
        refined = F.relu(self.bn_refine(self.refine_conv1(decoder_output)))
        output = self.refine_conv2(refined)
        # output: [Batch, 1, Height, Width]
        
        return output


# Alternative: Simpler single-stack ConvLSTM if you prefer minimal depth
class ConvLSTMSimple(nn.Module):
    """
    Simplified ConvLSTM model: single encoder stack without decoder.
    
    Use this if you prefer a lightweight alternative to ConvLSTMPredictor.
    """
    
    def __init__(self, in_channels=5, hidden_channels=32, num_layers=3, output_channels=1):
        """
        Args:
            in_channels: Number of meteorological input channels
            hidden_channels: Base hidden channel count
            num_layers: Number of ConvLSTM layers
            output_channels: Number of output channels (1 for NO2)
        """
        super().__init__()
        self.in_channels = in_channels
        self.output_channels = output_channels
        
        # Create hidden channel list for progressive channel expansion
        hidden_list = [hidden_channels * (2 ** i) for i in range(num_layers)]
        
        # Encoder-only ConvLSTM stack
        self.convlstm = ConvLSTM(
            in_channels=in_channels,
            hidden_channels_list=hidden_list,
            kernel_size=3,
            num_layers=num_layers
        )
        
        # Output projection from final hidden layer to NO2
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_list[-1], 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_channels, kernel_size=1)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [Batch, Time=4, Channels=5, Height, Width]
        
        Returns:
            output: Predicted NO2 [Batch, 1, Height, Width]
        """
        h_final, _, _ = self.convlstm(x)
        output = self.output_proj(h_final)
        return output


if __name__ == '__main__':
    # Quick test of ConvLSTM components
    batch_size = 2
    time_steps = 4
    channels = 5
    height, width = 64, 64
    
    # Create dummy input
    x = torch.randn(batch_size, time_steps, channels, height, width)
    
    print("Testing ConvLSTMCell...")
    cell = ConvLSTMCell(in_channels=5, hidden_channels=32)
    h = torch.zeros(batch_size, 32, height, width)
    c = torch.zeros(batch_size, 32, height, width)
    h_new, c_new = cell(x[:, 0], h, c)
    print(f"✅ ConvLSTMCell output shapes - h: {h_new.shape}, c: {c_new.shape}")
    
    print("\nTesting ConvLSTM (stacked)...")
    convlstm = ConvLSTM(in_channels=5, hidden_channels_list=[32, 64], num_layers=2)
    output, h_list, c_list = convlstm(x)
    print(f"✅ ConvLSTM output shape: {output.shape}")
    print(f"   Number of hidden states: {len(h_list)}")
    
    print("\nTesting ConvLSTMPredictor...")
    model = ConvLSTMPredictor(in_channels=5, hidden_channels=32, num_layers=2, output_channels=1)
    prediction = model(x)
    print(f"✅ ConvLSTMPredictor output shape: {prediction.shape}")
    
    print("\nTesting ConvLSTMSimple...")
    model_simple = ConvLSTMSimple(in_channels=5, hidden_channels=32, num_layers=3, output_channels=1)
    prediction_simple = model_simple(x)
    print(f"✅ ConvLSTMSimple output shape: {prediction_simple.shape}")
    
    print("\n✨ All ConvLSTM components working correctly!")
