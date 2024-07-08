from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model

def build_resnet_model(input_shape, num_classes):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    return model

input_shape = (224, 224, 3)  # Adjust based on your patches
num_classes = 2  # Replace with your number of classes

model = build_resnet_model(input_shape, num_classes)
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])