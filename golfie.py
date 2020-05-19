import vision
import learning
import serial
import numpy as np
import pandas as pd


def main():
	#######  Estimate Distance  ########
	vision.img_capture()
	distance = vision.img_distance()
	print("Distance estimated ", distance)
	if distance == 0:
		print("Distance estimation Failed! ")
		exit(0)
	#######  Get KNN Model  ########
	knn_model = learning.learn()

	#######  Predict hit Success/Failure  ########
	d = distance
	temp = {'Distance': [d, d, d, d, d], 'Delay': [1, 2, 3, 4, 5]}
	x1 = pd.DataFrame(data=temp)
	y_pred = learning.knn_predict(knn_model, x1)
	print("Prediction successful!")
	print(y_pred)

	idx = np.where(y_pred == np.amax(y_pred))
	print(idx[0])
	idx = np.amax(idx)
	print(idx)

	delay = idx+1
	print('Delay to be applied: ', delay)


	#######  Execute hit  ########
	ser = serial.Serial()
	ser.baudrate = 9600
	ser.port = 'com5'
	ser.open()
	i = bytearray([delay])
	#send value here
	ser.write(i)
	print("sent")
	ser.close()


if __name__ == "__main__":
	main()