import cv2
import gzip
import pickle


with gzip.open('s2_mnist.gz', 'rb') as f:
    data = pickle.load(f)

raw_imgs = data['train']['raw_images']
proj_imgs = data['train']['images']

for raw_img, proj_img in zip(raw_imgs, proj_imgs):
    cv2.imshow("raw_img", raw_img)
    cv2.imshow("proj_img", proj_img)
    cv2.waitKey(0)
