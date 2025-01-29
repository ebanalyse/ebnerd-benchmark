from ebrec.models.newsrec.model_config import hparams_lstur_docvec
from ebrec.models.newsrec.lstur_docvec import LSTURDocVec
import numpy as np

config = hparams_lstur_docvec

DOCVEC_DIM = 300
BATCH_SIZE = 10
HISTORY_SIZE = 20
NPRATIO = 4

#
config.history_size = HISTORY_SIZE
config.title_size = DOCVEC_DIM
config.newsencoder_units_per_layer = [256, 256]

# MODEL:
model = LSTURDocVec(hparams=config)
model.model.summary()

#
his_input_title_shape = (HISTORY_SIZE, DOCVEC_DIM)
pred_input_title_shape = (NPRATIO + 1, DOCVEC_DIM)
label_shape = (NPRATIO + 1,)
n_users = config.n_users
user_indexes_shape = (1,)

# Generate some random input data for input_1
his_input_title = np.array(
    [np.random.rand(*his_input_title_shape) for _ in range(BATCH_SIZE)]
)
# Generate some random input data for input_2
pred_input_title = np.array(
    [np.random.rand(*pred_input_title_shape) for _ in range(BATCH_SIZE)]
)
# Input data for user_indexes
user_indexes = np.random.randint(0, n_users, size=(BATCH_SIZE, *user_indexes_shape))
# Generate some random label data with values between 0 and 1
label_data = np.zeros((BATCH_SIZE, *label_shape), dtype=int)
for row in label_data:
    row[np.random.choice(label_shape[0])] = 1

# Print the shapes of the input data to verify they match the model's input layers
print(his_input_title.shape)
print(pred_input_title.shape)
print(user_indexes.shape)
print(label_data.shape)

# Make input for model:
input = (user_indexes, his_input_title, pred_input_title)

# fit/predict:
model.model.fit(input, label_data, epochs=10)
model.model.predict(input)
