from ultralytics import YOLO


def train_model():
    model = YOLO("yolov8n.pt")

    model.train(
        data=r"F:\F\School\3\CS4391\Project\Data\combined_data\data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        task="detect",
    )


if __name__ == "__main__":
    train_model()