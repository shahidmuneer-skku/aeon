import torch
import torch.nn as nn     

class DeepHashNet(nn.Module):
   
    def __init__(self):
        super().__init__()
        self.flatten_dim = 4 * 64 * 64            # =16 384
        self.hash_layer  = nn.Linear(self.flatten_dim, self.flatten_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x_flat = x.view(B, -1)                    # 1. Flatten

        z = torch.tanh(self.hash_layer(x_flat))   # 2. Forward pass → Z

        # 3. Straight‑through estimator  H ← sign(Z) + (Z – stop‑grad(Z))
        hash_code = (z.sign() - z).detach() + z           # forward = sign(z), grad = 1
        
        reconstructed = hash_code.view(B, 4, 64, 64)  # Reshape to (B, 4, 64, 64)

        return reconstructed


hashnet = DeepHashNet()
hashes = hashnet(torch.randn((5,4,64,64)))
print(hashes.shape, hashes[:1])