import capsnet_model
import preprocessors
import capslayers
import data_preparer
import tensorflow as tf
import pickle

# importlib.reload(capslayers)
# importlib.reload(preprocessors)
# sess = tf.InteractiveSession()
model, eval_model = capsnet_model.main()

with open("./data/pickle/model1.pkl", "w+") as f: 
  pickle.dump([model, eval_model], f)

# import capslayers_my
# import data_preparer

# dp, input_shape = data_preparer.getDataProvider()

# train_dp, val_dp = dp.split()

# model = capslayers_my.CapsNet2(input_shape, len(data_preparer.VOCAB) + 1, len(data_preparer.VOCAB) + 1)