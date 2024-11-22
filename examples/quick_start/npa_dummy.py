# TODO make a notebook with it
from ebrec.models.newsrec.model_config import hparams_npa
from ebrec.models.newsrec.npa import NPAModel
import numpy as np

config = hparams_npa

# Define the number of samples in your batch
BATCH_SIZE = 300
HISTORY_SIZE = config.history_size
TITLE_SIZE = config.title_size
NPRATIO = 4
word_embeddings = np.random.rand(1000, 100)

# Define the shapes of the input data
his_input_title_shape = (HISTORY_SIZE, TITLE_SIZE)
pred_input_title_shape = (NPRATIO + 1, TITLE_SIZE)
vocab_size = word_embeddings.shape[0]
n_users = config.n_users
label_shape = (NPRATIO + 1,)
user_indexes_shape = (1,)

model = NPAModel(hparams=config)
model.model.summary()

# Generate some random input data for input_1 with values between 0 and 1
his_input_title = np.random.randint(0, vocab_size, (BATCH_SIZE, *his_input_title_shape))
# Generate some random input data for input_2 with values between 0 and 1
pred_input_title = np.random.randint(
    0, vocab_size, (BATCH_SIZE, *pred_input_title_shape)
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
model.model.fit(input, label_data)
model.model.predict(input)
