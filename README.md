## Dataset Preparation

The dataset was collected from multiple online sources. Since the original datasets came from different folders and used different class structures, I wrote a Python script called `merge_yolo_datasets.py` to combine them into one YOLO-format dataset.

The script performs the following steps:

1. Reads multiple source datasets.
2. Copies images and label files into one combined dataset folder.
3. Renames files with prefixes to avoid duplicate filenames.
4. Remaps all labels into one unified class ID system.
5. Counts the number of images and labels in train, validation, and test sets.
6. Automatically creates the `data.yaml` file for YOLO training.

The final dataset contains six classes:

- soccer_ball
- keyboard
- cellphone
- laptop
- mouse
- pen