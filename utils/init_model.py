import torch
import segmentation_models_pytorch as smp
from ultralytics import YOLO


def init_model(device = "cuda:1"):
    checkpoint = "/home/godwinkhalko/ISRO/CodeBase/UNetPP/isro_unetplusplus_resnet34.pth"

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

def init_ROI_model(device):
    checkpoint = "/home/godwinkhalko/ISRO/CodeBase/ROI_Detector/runs/detect/runs/isro/exp1/weights/best.pt"
    model = YOLO(checkpoint)
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model, device