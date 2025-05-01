import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.nn import functional as F

# Hyperparameters
PATCH_SIZE = 14  # Split 28x2.8 into 2x2 patches
NUM_CLASSES = 10
EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

# 1. Data loading
transform = transforms.Compose([
    ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 2. Vision Transformer Model
class VisionTransformer(nn.Module):
    def __init__(self, num_patches, embed_dim=64, num_heads=2, num_layers=2):
        super(VisionTransformer, self).__init__()
        self.patch_size = PATCH_SIZE
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embedding = nn.Conv2d(1, embed_dim, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        self.class_token = nn.Parameter(torch.randn(1, embed_dim))  # Class token

        # Positional embeddings (learned)
        self.positional_encoding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        # Transformer encoder
        self.transformer = nn.ModuleDict({
            'layer_norm': nn.LayerNorm(embed_dim),
            'transformer': nn.ModuleList([
                nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
                for _ in range(num_layers)
            ])
        })

        # Final classification head
        self.classifier = nn.Linear(embed_dim, NUM_CLASSES)

    def forward(self, x):
        # Step 1: Patch embedding
        x = self.patch_embedding(x)  # Shape: [B, embed_dim, 2, 2]
        x = x.flatten(2)  # Shape: [B, embed_dim, 4]
        x = x.transpose(1, 2)  # Shape: [B, 4, embed_dim]

        # Step 2: Add class token
        class_token = self.class_token.repeat(x.size(0), 1)
        x = torch.cat([class_token.unsqueeze(1), x], dim=1)  # Shape: [B, 5, embed_dim]

        # Step 3: Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]

        # Step 4: Transformer encoder
        for attn in self.transformer['transformer']:
            x = attn(x, x, x)

        # Step 5: Classification
        x = self.transformer['layer_norm'](x)
        return self.classifier(x[:, 0])  # Use class token for classification

# 3. Initialize model
num_patches = (28 // PATCH_SIZE) ** 2  # 2x2 = 4 patches
model = VisionTransformer(num_patches=num_patches, embed_dim=64, num_heads=2, num_layers=2)

# 4. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 5. Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# 6. Test (optional)
# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#     print(f"Accuracy: {100 * correct / total:.2f}%")