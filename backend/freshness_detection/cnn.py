import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("/Users/sahitipotini/Desktop/food_detect_and_freshness_api/freshness3.h5", compile=False)

# Summary of architecture
model.summary()

# Check input shape
print("Input shape:", model.input_shape)

# Check output shape
print("Output shape:", model.output_shape)

# Look at layer details
for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.__class__.__name__, layer.output_shape)
