import numpy as np
from keras.applications import vgg16
from keras.preprocessing import image
from keras.activations import relu, softmax
import keras.backend as K
import matplotlib.pyplot as plt

from keras import backend as K
# set GPU memory 
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
def color_preprocessing(x_train):
    x_train = x_train.astype('float32')
    mean = [125.3, 123.0, 113.9]
    std  = [63.0,  62.1,  66.7]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
    return x_train

def scheduler(epoch):
    lr = 0.0008
    if epoch < 60:
        lr= 0.1
    elif epoch < 120:
        lr= 0.02
    elif epoch < 160:
        lr= 0.004
    else:
        lr = 0.0008
    return lr



def residual_block(x,out_filters,increase=False):
    global IN_FILTERS
    stride = (1,1)
    if increase:
        stride = (2,2)
        
    o1 = bn_relu(x)
    
    conv_1 = Conv2D(out_filters,
        kernel_size=(3,3),strides=stride,padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(WEIGHT_DECAY),
        use_bias=False)(o1)

    o2 = bn_relu(conv_1)
    
    conv_2 = Conv2D(out_filters, 
        kernel_size=(3,3), strides=(1,1), padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(WEIGHT_DECAY),
        use_bias=False)(o2)
    if increase or IN_FILTERS != out_filters:
        proj = Conv2D(out_filters,
                            kernel_size=(1,1),strides=stride,padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(WEIGHT_DECAY),
                            use_bias=False)(o1)
        block = add([conv_2, proj])
    else:
        block = add([conv_2,x])
    return block


def bn_relu(x):
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    return x


   
def conv3x3(x,filters):
    return Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), padding='same',
    kernel_initializer='he_normal',
    kernel_regularizer=l2(WEIGHT_DECAY),
    use_bias=False)(x)



def resnet(img_input,classes_num,depth,k):
    n_filters  = [16, 16*k, 32*k, 64*k]
    n_stack    = (depth - 4) // 6

    def wide_residual_layer(x,out_filters,increase=False):
        global IN_FILTERS
        x = residual_block(x,out_filters,increase)
        IN_FILTERS = out_filters
        for _ in range(1,int(n_stack)):
            x = residual_block(x,out_filters)
        return x

    x = conv3x3(img_input,n_filters[0])
    x = wide_residual_layer(x,n_filters[1])
    x = wide_residual_layer(x,n_filters[2],increase=True)
    x = wide_residual_layer(x,n_filters[3],increase=True)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((8,8))(x)
    x = Flatten()(x)
    x = Dense(classes_num,
        activation='softmax',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(WEIGHT_DECAY),
        use_bias=False)(x)
    return x

img_input = Input(shape=(32,32,3))
output = resnet(img_input,10,28,10)
net = Model(img_input, output)

net.load_weights('Resnet_weights.h5')


img_path = 'angry.jpg'
img = image.load_img(img_path)

plt.imshow(img)
plt.grid('off')
plt.axis('off')

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)


# Get the initial predictions
preds = model.predict(x)
initial_class = np.argmax(preds)


# Inverse of the preprocessing and plot the image
def plot_img(x):
    """
    x is a BGR image with shape (? ,224, 224, 3) 
    """
    t = np.zeros_like(x[0])
    t[:,:,0] = x[0][:,:,2]
    t[:,:,1] = x[0][:,:,1]
    t[:,:,2] = x[0][:,:,0]  
    plt.imshow(np.clip((t+[123.68, 116.779, 103.939]), 0, 255)/255)
    plt.grid('off')
    plt.axis('off')
    plt.show()

# Sanity Check
plot_img(x)

# Get current session (assuming tf backend)
sess = K.get_session()
# Initialize adversarial example with input image
x_adv = x
# Added noise
x_noise = np.zeros_like(x)

# Set variables
epochs = 400
epsilon = 0.01
target_class = 0 
prev_probs = []

for i in range(epochs): 
    # One hot encode the target class
    target = K.one_hot(target_class, 10)
    
    # Get the loss and gradient of the loss wrt the inputs
    loss = -1*K.categorical_crossentropy(target, model.output)
    grads = K.gradients(loss, model.input)

    # Get the sign of the gradient
    delta = K.sign(grads[0])
    x_noise = x_noise + delta

    # Perturb the image
    x_adv = x_adv + epsilon*delta

    # Get the new image and predictions
    x_adv = sess.run(x_adv, feed_dict={model.input:x})
    preds = model.predict(x_adv)

    # Store the probability of the target class
    prev_probs.append(preds[0][target_class])


plot_img(x_adv)
plot_img(x_adv-x)