# TODO make a notebook with it
from ebrec.models.newsrec.model_config import hparams_naml
from ebrec.models.newsrec.naml import NAMLModel
import numpy as np

config = hparams_naml

# Define the number of samples in your batch
BATCH_SIZE = 300
NPRATIO = 4
HISTORY_SIZE = config.history_size
TITLE_SIZE = config.title_size
BODY_SIZE = config.body_size

label_shape = (NPRATIO + 1,)
word_embeddings = np.random.rand(1000, 100)

vocab_size = word_embeddings.shape[0]
n_verts = config.vert_num
n_subverts = config.subvert_num

# Model
model = NAMLModel(hparams=config, word2vec_embedding=word_embeddings)
model.model.summary()

# Define the shapes of the input data
his_input_title = np.random.randint(
    0, vocab_size, size=(BATCH_SIZE, HISTORY_SIZE, TITLE_SIZE)
)
his_input_body = np.random.randint(
    0, vocab_size, size=(BATCH_SIZE, HISTORY_SIZE, BODY_SIZE)
)
his_input_vert = np.random.randint(0, n_verts, size=(BATCH_SIZE, HISTORY_SIZE, 1))
his_input_subvert = np.random.randint(0, n_subverts, size=(BATCH_SIZE, HISTORY_SIZE, 1))
pred_input_title = np.random.randint(
    0, vocab_size, size=(BATCH_SIZE, NPRATIO + 1, TITLE_SIZE)
)
pred_input_body = np.random.randint(
    0, vocab_size, size=(BATCH_SIZE, NPRATIO + 1, BODY_SIZE)
)
pred_input_vert = np.random.randint(0, n_verts, size=(BATCH_SIZE, NPRATIO + 1, 1))
pred_input_subvert = np.random.randint(0, n_subverts, size=(BATCH_SIZE, NPRATIO + 1, 1))

# Generate some random label data with values between 0 and 1
label_data = np.zeros((BATCH_SIZE, *label_shape), dtype=int)
for row in label_data:
    row[np.random.choice(label_shape[0])] = 1

#
his_input_title.shape
his_input_body.shape
his_input_vert.shape
his_input_subvert.shape
pred_input_title.shape
pred_input_body.shape
pred_input_vert.shape
pred_input_subvert.shape
label_data.shape

# Make input for model:
input = (
    his_input_title,
    his_input_body,
    his_input_vert,
    his_input_subvert,
    pred_input_title,
    pred_input_body,
    pred_input_vert,
    pred_input_subvert,
)

# fit/predict:
model.model.fit(input, label_data)
model.model.predict(input)
