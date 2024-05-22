# TODO make a notebook with it
from ebrec.models.newsrec.model_config import hparams_nrms
from ebrec.models.newsrec.nrms import NRMSModel
import numpy as np

config = hparams_nrms

# Define the number of samples in your batch
BATCH_SIZE = 10
HISTORY_SIZE = config.history_size
TITLE_SIZE = config.title_size
NPRATIO = 4
word_embeddings = np.random.rand(1000, 100)

model = NRMSModel(hparams=config, word2vec_embedding=word_embeddings)
model.model.summary()

# Define the shapes of the input data
his_input_title_shape = (HISTORY_SIZE, TITLE_SIZE)
pred_input_title_shape = (NPRATIO + 1, TITLE_SIZE)
label_shape = (NPRATIO + 1,)
vocab_size = word_embeddings.shape[0]

# Generate some random input data for input_1 with values between 0 and 1
his_input_title = np.random.randint(0, vocab_size, (BATCH_SIZE, *his_input_title_shape))

# Generate some random input data for input_2 with values between 0 and 1
pred_input_title = np.random.randint(
    0, vocab_size, (BATCH_SIZE, *pred_input_title_shape)
)

# Generate some random label data with values between 0 and 1
label_data = np.zeros((BATCH_SIZE, *label_shape), dtype=int)
for row in label_data:
    row[np.random.choice(label_shape[0])] = 1

# Print the shapes of the input data to verify they match the model's input layers
print(his_input_title.shape)
print(pred_input_title.shape)
print(label_data.shape)

# Make input for model:
input = (his_input_title, pred_input_title)

# fit/predict:
model.model.fit(input, label_data)
model.model.predict(input)
