from ultralytics import YOLO


def camera_tracking():
    model = YOLO(r"runs\detect\train-7\weights\best.pt")

    model.track(
        source=0,
        imgsz=640,
        conf=0.25,
        show=True,
        save=False,
        persist=True
    )


if __name__ == "__main__":
    camera_tracking()