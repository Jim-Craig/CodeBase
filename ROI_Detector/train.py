from ultralytics import YOLO


if __name__ == "__main__":
    # Load the YOLOv8 model (you can specify a pretrained model or start from scratch)
    model = YOLO("yolov8n.pt")  # Use 'yolov8n.pt' for a small model, or 'yolov8m.pt' for medium, etc.

    # Train the model on your dataset
    model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device="cuda",      # or "cpu" / "mps" for Apple Silicon
    project="runs/isro",
    name="exp1",
    patience=20,        # early stopping
)