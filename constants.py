import helper

# Runtime parameters
batch_size = 32
epochs = 50
img_size = 64

# Dataset location
training_set = 'dataset/train'
test_set = 'dataset/test'

steps_per_epoch = helper.stepCount(training_set, batch_size)
validation_steps = helper.stepCount(test_set, batch_size)

# Training model filenames
model_json = "model.json"
model_weight = "model.h5"