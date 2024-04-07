import os
import pandas as pd
import cv2

#### FOR TESTING AND TRAINING DATASET ####

for filename in os.listdir('train'):
    f = os.path.join('train', filename)
    
    if os.path.isfile(f):
      img = cv2.imread(f)
      if img is None:
        continue
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

      # Loading Haar Cascade model
      face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
      faces = face_cascade.detectMultiScale(gray, 1.1, 4)

      if len(faces) == 0:
        path = os.path.join('cropped_train')
        cv2.imwrite(os.path.join(path, filename), img)
      else:
        # Crop and save the first detected face
        x = faces[0][0]
        y = faces[0][1]
        w = faces[0][2]
        h = faces[0][3]
        cropped_face = img[y:y + h, x:x + w]
        path = os.path.join('cropped_train')
        cv2.imwrite(os.path.join(path, filename), cropped_face)



#### FOR TRAINING DATASET ONLY ####

base_dir = 'cropped_folders' # Folder name to store train data after processing
train_files = 'cropped_train' # Same path as in line 30, to use images after face detection

df = pd.read_csv('train.csv')

for index, row in df.iterrows():
    img_path = os.path.join(train_files, row['File Name'])
    celebrity_name = row['Category']

    target_dir = os.path.join(base_dir, celebrity_name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    target_path = os.path.join(target_dir, row['File Name'])

    image = cv2.imread(img_path)

    if image is not None:
        cv2.imwrite(target_path, image)
        print(f"saved {img_path} to {target_path}")
    else:
        print(f"No face in {img_path}")