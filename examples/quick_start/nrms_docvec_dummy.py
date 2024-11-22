# TODO make a notebook with it
from ebrec.models.newsrec.nrms_docvec import NRMSDocVec
from ebrec.models.newsrec.model_config import hparams_nrms
import numpy as np

DOCVEC_DIM = 300
BATCH_SIZE = 10
HISTORY_SIZE = 20
NPRATIO = 4

#
config = hparams_nrms
config.history_size = HISTORY_SIZE
config.title_size = DOCVEC_DIM

# MODEL:
model = NRMSDocVec(hparams=config, newsencoder_units_per_layer=[512, 512])
model.model.summary()

#
his_input_title_shape = (HISTORY_SIZE, DOCVEC_DIM)
pred_input_title_shape = (NPRATIO + 1, DOCVEC_DIM)
label_shape = (NPRATIO + 1,)

# Generate some random input data for input_1
his_input_title = np.array(
    [np.random.rand(*his_input_title_shape) for _ in range(BATCH_SIZE)]
)
# Generate some random input data for input_2
pred_input_title = np.array(
    [np.random.rand(*pred_input_title_shape) for _ in range(BATCH_SIZE)]
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
model.model.fit(input, label_data, epochs=10)
model.model.predict(input)
