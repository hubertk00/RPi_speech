from keras.api import layers, models, Input, regularizers
from models.models_wake_word import residual_block_type_one, residual_block_type_two

def resnet8(input_shape, num_classes, k):
    input_layer = layers.Input(shape=input_shape)
    
    x = layers.Conv1D(
        int(16*k), 1, 
        strides=1, 
        use_bias=False,
        padding='same',
        kernel_regularizer=regularizers.l2(0.001) 
    )(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)  
    
    x = residual_block_type_two(x, 24, k)
    x = layers.Dropout(0.15)(x)
    
    x = residual_block_type_two(x, 32, k)
    x = layers.Dropout(0.2)(x)  
    
    x = residual_block_type_two(x, 48, k)
    x = layers.Dropout(0.25)(x) 
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Flatten()(x)
    
    x = layers.Dense(
        int(64 * k), 
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001)
    )(x)
    x = layers.Dropout(0.4)(x)  
    
    output = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=input_layer, outputs=output)

def resnet14(input_shape, num_classes, k):
    input_layer = layers.Input(shape=input_shape)
    
    x = layers.Conv1D(
        int(16 * k), 3, 
        strides=1, 
        use_bias=False, 
        padding='same',
        kernel_regularizer=regularizers.l2(0.001)
    )(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    x = residual_block_type_two(x, 24, k)
    x = layers.Dropout(0.15)(x) 
    
    x = residual_block_type_one(x, 24, k)
    x = layers.Dropout(0.15)(x) 
    
    x = residual_block_type_two(x, 32, k)
    x = layers.Dropout(0.2)(x)
    
    x = residual_block_type_one(x, 32, k)
    x = layers.Dropout(0.2)(x) 
    
    x = residual_block_type_two(x, 48, k)
    x = layers.Dropout(0.25)(x) 
    
    x = residual_block_type_one(x, 48, k)
    x = layers.Dropout(0.25)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Flatten()(x)
    
    x = layers.Dense(
        int(128 * k), 
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001)
    )(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(
        int(64 * k), 
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001)
    )(x)
    x = layers.Dropout(0.4)(x)
    
    output = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=input_layer, outputs=output)

def crnn(input_shape, num_classes, k):
    input_layer = layers.Input(shape=input_shape)

    x = layers.Conv1D(
        filters=int(32 * k),
        kernel_size=3,
        strides=1,
        padding='causal',
        use_bias=False,
        kernel_regularizer=regularizers.l2(0.001)
    )(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.1)(x) 

    x = layers.GRU(
        units=int(64 * k),
        return_sequences=False,
        kernel_regularizer=regularizers.l2(0.001),
        recurrent_dropout=0.15 
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x) 

    x = layers.Dense(
        units=int(32 * k),
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001)
    )(x)
    x = layers.Dropout(0.4)(x)

    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs=input_layer, outputs=output_layer)