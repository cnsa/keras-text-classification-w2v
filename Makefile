tensorboard:
	tensorboard --logdir ./Graph/

model:
	python ./model.py

eval:
	python ./eval.py
