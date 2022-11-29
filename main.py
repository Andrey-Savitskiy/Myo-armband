import os

import pygame
from pynput.keyboard import Key, Controller
from pyomyo import emg_mode
from Classifier import Live_Classifier, MyoClassifier, EMGHandler
from xgboost import XGBClassifier


DINO_MODE = True


def dino_handler(pose):
	print("Pose detected", pose)
	if ((pose == 1) and (DINO_MODE)):
		keyboard = Controller()
		for i in range(0,10):
			# Press and release space
			keyboard.press(Key.space)
			keyboard.release(Key.space)


def get_user_name():
	name = input('\nВведите имя пользователя: ')
	path = f'data/{name}'
	if os.path.exists(path):
		print(f'Данные пользователя {name} успешно загружены.\n\n')
	else:
		print(f'Создан новый пакет данных для пользователя {name}.\n\n')

	return name


def main():
	username = get_user_name()

	pygame.init()
	w, h = 800, 320
	scr = pygame.display.set_mode((w, h))
	font = pygame.font.Font(None, 30)

	# Make an ML Model to train and test with live
	# XGBoost Classifier Example
	model = XGBClassifier(eval_metric='logloss')
	clr = Live_Classifier(model, name="XG", color=(0,102,51), user_name=username)
	m = MyoClassifier(clr, mode=emg_mode.PREPROCESSED, hist_len=10)

	hnd = EMGHandler(m)
	m.add_emg_handler(hnd)
	m.connect()

	m.add_raw_pose_handler(dino_handler)

	# Set Myo LED color to model color
	m.set_leds((255, 0, 0), (255, 255, 0))
	# Set pygame window name
	pygame.display.set_caption(m.cls.name)

	try:
		while True:
			# Run the Myo, get more data
			m.run()
			# Run the classifier GUI
			m.run_gui(hnd, scr, font, w, h)

	except KeyboardInterrupt:
		pass
	finally:
		m.disconnect()
		print()
		pygame.quit()


if __name__ == '__main__':
	main()