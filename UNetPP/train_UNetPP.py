import argparse
import os
import random
import torchvision
import torchvision.transforms.functional as TF
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

#Dataset Class
class ISRODataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform=None, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augment = augment

        # Sort both lists to guarantee alignment between image and mask
        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir) if f.endswith('.tiff')
        ])
        self.mask_paths = sorted([
            os.path.join(mask_dir, f)
            for f in os.listdir(mask_dir) if f.endswith('.png')
        ])

        assert len(self.image_paths) == len(self.mask_paths), \
            f"Mismatch: {len(self.image_paths)} images vs {len(self.mask_paths)} masks"

    def __len__(self):
        return len(self.image_paths)

    def apply_shared_augmentations(self, image, mask):
        """
        Spatial augmentations must be applied IDENTICALLY to both
        image and mask. We use TF (transforms.functional) with a
        shared random seed so both get the exact same transformation.
        """
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        return image, mask

    def __getitem__(self, idx):
        # --- Load ---
        image = Image.open(self.image_paths[idx]).convert('L')  # Grayscale
        mask = Image.open(self.mask_paths[idx]).convert('L')    # Grayscale

        # --- Resize both (spatial, must match) ---
        resize = torchvision.transforms.Resize((256, 256))
        image = resize(image)
        mask = resize(mask)

        # --- Synchronized spatial augmentations ---
        if self.augment:
            image, mask = self.apply_shared_augmentations(image, mask)

        # --- Image-only augmentations (never apply to mask) ---
        if self.augment:
            if random.random() > 0.5:
                image = TF.gaussian_blur(image, kernel_size=[5, 9], sigma=[0.1, 5.0])
            if random.random() > 0.5:
                # ColorJitter contrast only — safe for grayscale X-rays
                image = torchvision.transforms.ColorJitter(contrast=0.5)(image)

        # --- To tensor ---
        image = TF.to_tensor(image)   # Shape: [1, H, W], values in [0, 1]
        mask = TF.to_tensor(mask)     # Shape: [1, H, W], values in [0, 1]

        # --- Normalize image ONLY, after all augmentations ---
        image = TF.normalize(image, mean=[0.485], std=[0.229])

        # --- Binarize mask: make it strictly 0 or 1 ---
        mask = (mask > 0.5).float()   # [1, H, W] with values {0.0, 1.0}

        return image, mask
    
#Loss functions
def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1e-6
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

#Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()
    
#loss function
def loss_fn(pred, target):
    bce = nn.BCEWithLogitsLoss()
    return bce(pred, target) + dice_loss(pred, target)

# Checkpoiting function
def save_checkpoint(model, optimizer, epoch, loss, save_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, save_path)

#Training loop
def train_model(model, train_loader, val_loader, optimizer, checkpoint_path, num_epochs=10, device='cuda:0'):
    best_val_loss = float('inf')
    best_model = model
    best_epoch = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        print(f"Epoch {epoch+1}/{num_epochs} - Training...")
        for idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print(f"Batch Loss for batch {idx}: {loss.item():.4f}", end='\r')
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        print(f"Epoch {epoch+1}/{num_epochs} - Validating...")
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        # Print training and validation loss for the epoch
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint after each epoch if validation loss improved, 
        # stop the training if validation loss does not improve for 5 consecutive epochs
        if epoch == 0 or avg_val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f} on epoch {epoch+1}. Saving checkpoint.")
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_model = model
            save_checkpoint(model, optimizer, epoch+1, avg_val_loss, checkpoint_path)
        else:
            if epoch - best_epoch >= 5:
                print("Validation loss has not improved for 5 consecutive epochs. Stopping training.")
                break
    return best_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #take command line arguments for hyperparameters and paths
    parser.add_argument('--train_image_dir', type=str, default='/home/godwinkhalko/ISRO/Data/train_images', help='Path to training images')
    parser.add_argument('--train_mask_dir', type=str, default='/home/godwinkhalko/ISRO/Data/train_masks', help='Path to training masks')
    parser.add_argument('--val_image_dir', type=str, default='/home/godwinkhalko/ISRO/Data/val_images', help='Path to validation images')
    parser.add_argument('--val_mask_dir', type=str, default='/home/godwinkhalko/ISRO/Data/val_masks', help='Path to validation masks')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--save_path', type=str, default='/home/godwinkhalko/ISRO/UNetPP/isro_unetplusplus_resnet34.pth', help='Path to save the trained model')
    parser.add_argument('--checkpoint_path', type=str, default='/home/godwinkhalko/ISRO/UNetPP/checkpoint.pth', help='Path to save the checkpoint')
    parser.add_argument('--resume_checkpoint', action='store_true', help='Whether to resume training from checkpoint')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training (e.g., "cuda:0" or "cpu")')
    args = parser.parse_args()

    
    train_dataset = ISRODataset(
        image_dir=args.train_image_dir,
        mask_dir=args.train_mask_dir,
        augment=True      # augment=False for val/test
    )

    val_dataset = ISRODataset(
        image_dir=args.val_image_dir,
        mask_dir=args.val_mask_dir,
        augment=False
    )    

    #Init Model
    model = smp.UnetPlusPlus(
    encoder_name="resnet34",   # backbone
    encoder_weights="imagenet",  # pretrained
    in_channels=1,  # X-ray = grayscale
    classes=1       # binary segmentation
    )
    if args.resume_checkpoint and os.path.exists(args.checkpoint_path):
        print(f"Resuming training from checkpoint: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Resumed model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")

    #Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    #Start Training
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model = train_model(model, train_loader, val_loader, optimizer, args.checkpoint_path, num_epochs=args.epochs, device=device)

    #Save the trained model
    print(f"Training complete. Saving best model to {args.save_path}")
    save_path = args.save_path
    torch.save(model.state_dict(), save_path)
    print(f"Best model saved to {save_path}")
    #clear GPUs
    torch.cuda.empty_cache()
    del model
    del optimizer
    del train_loader
    del val_loader
    del train_dataset
    del val_dataset
    