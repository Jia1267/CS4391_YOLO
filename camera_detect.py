from ultralytics import YOLO


def camera_detection():
    model = YOLO(r"runs\detect\train-7\weights\best.pt")

    model.predict(
        source=0,
        imgsz=640,
        conf=0.25,
        show=True,
        save=False
    )


if __name__ == "__main__":
    camera_detection()