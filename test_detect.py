from ultralytics import YOLO


def test_detection():
    model = YOLO(r"runs\detect\train-7\weights\best.pt")

    model.predict(
        source=r"F:\F\School\3\CS4391\Project\Data\combined_data\test\images",
        imgsz=640,
        conf=0.25,
        save=True
    )


if __name__ == "__main__":
    test_detection()