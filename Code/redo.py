import cv2
import pandas as pd
import numpy as np
import os

import pickle
train_df = pd.read_csv('petfinder-adoption-prediction/train/train.csv')
img_size = 256
batch_size = 1000
print(train_df)
pet_ids = train_df['PetID'].values
n_batches = len(pet_ids) // batch_size + 1

def resize_to_square(im, img_size):
	old_size = im.shape[:2] # old_size is in (height, width) format
	ratio = float(img_size)/max(old_size)
	new_size = tuple([int(x*ratio) for x in old_size])
	# new_size should be in (width, height) format
	im = cv2.resize(im, (new_size[1], new_size[0]))
	delta_w = img_size - new_size[1]
	delta_h = img_size - new_size[0]
	top, bottom = delta_h//2, delta_h-(delta_h//2)
	left, right = delta_w//2, delta_w-(delta_w//2)
	color = [0, 0, 0]
	new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
	return new_im

def load_image(path, pet_id, img_size):
	# print(f'{path}{pet_id}-1.jpg')
	image = cv2.imread(f'{path}{pet_id}-1.jpg')
	new_image = resize_to_square(image, img_size)
	new_image = preprocess_input(new_image)
	return new_image

features = {}
print(n_batches)
labels = []
arrays = []
cnt = 0
for b in range(n_batches):
	start = b*batch_size
	end = (b+1)*batch_size
	batch_pets = pet_ids[start:end]
	batch_images = np.zeros((len(batch_pets),img_size,img_size,3))
	for i,pet_id in enumerate(batch_pets):
		try:
			batch_images[i] = load_image("petfinder-adoption-prediction/train_images/", pet_id, img_size)
		except:
			pass
	arrays = []
	for i in range(batch_images.shape[0]):
		arrays.append(batch_images[i,:,:,:])
	with open('image' + str(b) + '.pkl', 'wb') as f:
		pickle.dump(arrays, f)
	# batch_preds = m.predict(batch_images)
	# print(batch_preds)
	# print(str(b) + " done")
	# for i,pet_id in enumerate(batch_pets):
	# 	features[pet_id] = batch_preds[i]

val = train_df['AdoptionSpeed'].tolist()
with open('label.pkl', 'wb') as f:
	pickle.dump(val, f)