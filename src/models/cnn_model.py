import torch
import torch.nn as nn


class TrafficCNN(nn.Module):
    """1D CNN for classifying flow-based network traffic features."""

    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 1,
        conv_channels: tuple[int, int] = (32, 64),
        kernel_size: int = 3,
        pool_kernel_size: int = 2,
        adaptive_pool_size: int = 4,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if len(conv_channels) < 2:
            raise ValueError("conv_channels must define at least two convolutional layers.")
        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must define at least one fully connected layer.")

        padding = kernel_size // 2

        self.feature_extractor = nn.Sequential(
            # First convolution block: learns local feature patterns across the 1D flow sequence.
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=conv_channels[0],
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(inplace=True),
            # First pooling layer: reduces sequence length while retaining dominant responses.
            nn.MaxPool1d(kernel_size=pool_kernel_size),

            # Second convolution block: captures higher-level interactions between learned features.
            nn.Conv1d(
                in_channels=conv_channels[0],
                out_channels=conv_channels[1],
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(inplace=True),
            # Second pooling layer: further compresses the sequence before classification.
            nn.MaxPool1d(kernel_size=pool_kernel_size),

            # Keeps the classifier input size stable even if the feature vector length changes.
            nn.AdaptiveAvgPool1d(adaptive_pool_size),
        )

        classifier_layers: list[nn.Module] = []
        in_features = conv_channels[-1] * adaptive_pool_size

        for hidden_dim in hidden_dims:
            # Fully connected layers combine extracted convolutional features for classification.
            classifier_layers.append(nn.Linear(in_features, hidden_dim))
            classifier_layers.append(nn.ReLU(inplace=True))
            classifier_layers.append(nn.Dropout(p=dropout))
            in_features = hidden_dim

        # Final linear layer outputs logits for binary classification: VoIP vs non-VoIP.
        classifier_layers.append(nn.Linear(in_features, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, num_features) or
               (batch_size, input_channels, sequence_length).

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        if x.dim() == 2:
            # Convert a flat feature vector into a 1-channel sequence for Conv1d.
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(
                "Expected input shape (batch_size, num_features) or "
                "(batch_size, channels, sequence_length)."
            )

        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)
