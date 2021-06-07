import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

n_train = mnist.train.num_examples              # 55,000 images for training
n_validation = mnist.validation.num_examples    # 5,000 images for validation
n_test = mnist.test.num_examples                # 10,000 images for testing

n_input = 784   # input layer
n_hidden1 = 512 # 1st hidden layer      default 512
n_hidden2 = 256 # 2nd hidden layer      default 256
n_hidden3 = 128 # 3rd hidden layer      default 128
n_output = 10   # output layer with 0-9 digits

# Constant initial parameters AKA Hyperparameters
learning_rate = 1e-4
n_iterations = 1000
batch_size = 128
dropout = 0.5           # Represents a threshold at which we eliminate some units at random, prevents overfitting


# We will now create the TensorFlow Graph, this uses tensors, which is a datastructure like an array
# or list, but it can be initialized and manipulated as it is passed through the graph, updating
# as it learns

# placeholders that we will be feeding data into
# Specifies the size of the data that we are feeding
X = tf.placeholder("float", [None, n_input])  #None represents any amount, and we will be inputing an
                                              #undefined number of 784-pixel images
Y = tf.placeholder("float", [None, n_output]) #None also represents any amount where we will be inputing
                                              #any possible outputs with 10 possible classes (0-9) digits
keep_prob = tf.placeholder(tf.float32)        # This controls the dropout rate, it is a placeholder rather
                                              # than a hyperparameter, as we want it to change from .5 in
                                              # the training to 1.0 in the testing


# The parameters that the network updates in the training process are the weights and bias
# values, these need to be initialized with some value instead of an empty placeholder
weights = {
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
    'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
}

biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
    'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
}

# Layers in the network
# Each hidden layer will execute matriX multiplication(tf.matmul) on the previous layer's outputs and
# the current layer's weights, finally it adds bias to these values. the last hidden network
# will apply a dropout operation
layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])         #Layer 1
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])   #HLayer 1
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])   #HLayer 2
layer_drop = tf.nn.dropout(layer_3, keep_prob)                      #Hlayer 3 & Dropout layer
output_layer = tf.matmul(layer_3, weights['out']) + biases['out']   #output layer

# Now we want to define the loss function that we want to optimize
# A popular choice is cross entropy aka log-loss

# We will also be using the adam optimizer for gradient descent
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        labels=Y, logits=output_layer
        ))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# On to the training process we will print out our evaluation of accuracy and
# loss, as the iterations increase, we hope to see that accuracy increases as
# loss decreases
correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))    #Gives accuracy score
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializes the session for runnign the Graph
# We are optimizing the loss function through each iteration.
# We must minimize the difference between the predicted labels of the images
# and the true labels of the images
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# This process involves 4 steps:
#   1. Propagate values forward through the network
#   2. Compute the loss
#   3. Propagate values backward through the network
#   4. Update the parameters

# This code will feed a mini-batch of images through the network, and we then
# print the accuracy and loss for that batch, not the entire model.

for i in range(n_iterations):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={
        X: batch_x, Y: batch_y, keep_prob: dropout
        })

    # print loss and accuracy (per minibatch)
    if i % 100 == 0:
        minibatch_loss, minibatch_accuracy = sess.run(
            [cross_entropy, accuracy],
            feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0}
            )
        print(
            "Iteration",
            str(i),
            "\t| Loss =",
            str(minibatch_loss),
            "\t| Accuracy =",
            str(minibatch_accuracy)
            )

#This runs the session on the test image. Making a keep_prob dropout rate at 1.0 to ensure all
#Units are active in the testing process
test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})
print("\nAccuracy on test set:", test_accuracy)

img = np.invert(Image.open("test_img.png").convert('L')).ravel()
prediction = sess.run(tf.argmax(output_layer, 1), feed_dict={X:[img]})
print("Prediction for my image:", np.squeeze(prediction))
