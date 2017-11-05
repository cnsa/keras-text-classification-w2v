tensorboard:
	tensorboard --logdir ./Graph/

model:
	python ./model.py

download:
	python ./webhose.py

eval:
	python ./eval.py
