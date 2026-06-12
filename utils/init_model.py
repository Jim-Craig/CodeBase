import torch
import segmentation_models_pytorch as smp
from ultralytics import YOLO

# This function initializes the UNet++ model for defect segmentation with the checkpoitn weight
#  sets them to evaluation mode. 
# Incase you want to change the model or the checkpoint, you can do it here.
# Make sure the model accepts grayscale images (in_channels=1) and outputs a single channel mask (classes=1) f
# or binary segmentation and the model accepts the images of shape 256x256 as input, 
# since the defect description code is designed for that input size.
#  If you change the model architecture, make sure to adjust the input and output channels accordingly.
def init_model(device = "cuda:1"):
    checkpoint = "/home/godwinkhalko/ISRO/CodeBase/UNetPP/isro_unetplusplus_resnet34_DATA.pth"

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",   # backbone
        encoder_weights="imagenet",  # pretrained
        in_channels=1,  # X-ray = grayscale
        classes=1       # binary segmentation
        )
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model.to(device)
    model.eval()  # Set to evaluation mode
    #iterate through validation images and predict masks
    model.eval()

    return model, device
# This function initializes the YOLO model for ROI detection with the checkpoint weights and sets it to evaluation mode.
# Similar to the UNet++ model, if you want to change the model architecture or checkpoint, you can do it here.
# Though this model is nearly perfect with all metrics at approximatly 0.99 - 1.0 on the validation set,
# If you do need to change it, make sure the model is trained to detect the SGP and LGP regions correctly 
# and model accepts images of the shape 640x640 as input, since the defect description code is designed for that input size.
def init_ROI_model(device):
    checkpoint = "/home/godwinkhalko/ISRO/CodeBase/ROI_Detector/runs/detect/runs/isro/exp1/weights/best.pt"
    model = YOLO(checkpoint)
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model, device