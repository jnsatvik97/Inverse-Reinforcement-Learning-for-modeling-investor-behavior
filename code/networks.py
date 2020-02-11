import tensorflow as tf
from layers import *

def hidden2(vec_input, n_hidden1, n_hidden2, n_outputs, nonlinearity1, nonlinearity2):

    h1 = linear_layer(vec_input, n_hidden1, nonlinearity1, scope='fc1')
    h2 = linear_layer(h1, n_hidden2, nonlinearity2, scope='fc2')
    out = linear_layer(h2, n_outputs, nonlinearity=None, scope='out')

    return out


def r_net(state_input, action_input, f1=1, k1=5, f2=2, k2=3, n_fc3=8, n_fc4=4, n_fc5=4, d=15):
    
    action_input = tf.reshape(action_input, [-1,d,d,1])
    state_input = tf.reshape(state_input, [-1,d])
   
    conv1 = tf.contrib.layers.conv2d(inputs=action_input, num_outputs=f1, kernel_size=k1, stride=1, padding="SAME", activation_fn=tf.nn.relu, scope='conv1')
   
    conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=f2, kernel_size=k2, stride=1, padding="SAME", activation_fn=tf.nn.relu, scope='conv2')
    # flatten to vector
    conv2_flat = tf.reshape(conv2, [-1, f2*d*d]) 
   
    fc3 = tf.contrib.layers.fully_connected(inputs=conv2_flat, num_outputs=n_fc3, activation_fn=tf.nn.relu, scope='fc3') # batch_size x 64
    # for each sample in batch, concatenate conved action with state
    fc3_action = tf.concat([fc3, state_input], 1)
    # one more fc layer
    fc4 = tf.contrib.layers.fully_connected(inputs=fc3_action, num_outputs=n_fc4, activation_fn=tf.nn.relu, scope='fc4')
    
    out = tf.contrib.layers.fully_connected(inputs=fc4, num_outputs=1, activation_fn=tf.nn.tanh, scope='out')

    return out


def r_net_dropout_l1l2(state_input, action_input, f1=1, k1=5, f2=2, k2=3, n_fc3=8, n_fc4=4, n_fc5=4, d=15):
    
    action_input = tf.reshape(action_input, [-1,d,d,1])
    state_input = tf.reshape(state_input, [-1,d])
    
    conv1 = tf.contrib.layers.conv2d(inputs=action_input, num_outputs=f1, kernel_size=k1, stride=1, padding="SAME", activation_fn=tf.nn.relu, scope='conv1')
    
    conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=f2, kernel_size=k2, stride=1, padding="SAME", activation_fn=tf.nn.relu, scope='conv2')
    # flatten to vector
    conv2_flat = tf.reshape(conv2, [-1, f2*d*d]) # batch_size x (f2*d*d)
    # feed conv output to linear layer before combining with state vector
    fc3 = tf.contrib.layers.fully_connected(inputs=conv2_flat, num_outputs=n_fc3, activation_fn=tf.nn.relu, weights_regularizer=tf.contrib.layers.l1_l2_regularizer(), scope='fc3') # batch_size x 64
    dropout3 = tf.contrib.layers.dropout(fc3, keep_prob=0.4, scope='dropout3')
    # for each sample in batch, concatenate conved action with state
    fc3_action = tf.concat([dropout3, state_input], 1)
    # one more fc layer
    fc4 = tf.contrib.layers.fully_connected(inputs=fc3_action, num_outputs=n_fc4, activation_fn=tf.nn.relu, weights_regularizer=tf.contrib.layers.l1_l2_regularizer(), scope='fc4')
    dropout4 = tf.contrib.layers.dropout(fc4, keep_prob=0.4, scope='dropout4')
    
    out = tf.contrib.layers.fully_connected(inputs=dropout4, num_outputs=1, activation_fn=tf.nn.tanh, scope='out')

    return out


def r_net_l1l2(state_input, action_input, f1=1, k1=5, f2=2, k2=3, n_fc3=4, n_fc4=4, n_fc5=4, d=15):
    
    action_input = tf.reshape(action_input, [-1,d,d,1])
    state_input = tf.reshape(state_input, [-1,d])
    
    conv1 = tf.contrib.layers.conv2d(inputs=action_input, num_outputs=f1, kernel_size=k1, stride=1, padding="SAME", activation_fn=tf.nn.relu, scope='conv1')
    
    conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=f2, kernel_size=k2, stride=1, padding="SAME", activation_fn=tf.nn.relu, scope='conv2')
    # flatten to vector
    conv2_flat = tf.reshape(conv2, [-1, f2*d*d]) 
    fc3 = tf.contrib.layers.fully_connected(inputs=conv2_flat, num_outputs=n_fc3, activation_fn=tf.nn.relu, weights_regularizer=tf.contrib.layers.l1_l2_regularizer(), scope='fc3') # batch_size x 64
    # for each sample in batch, concatenate conved action with state
    fc3_action = tf.concat([fc3, state_input], 1)
    # one more fc layer
    fc4 = tf.contrib.layers.fully_connected(inputs=fc3_action, num_outputs=n_fc4, activation_fn=tf.nn.relu, weights_regularizer=tf.contrib.layers.l1_l2_regularizer(), scope='fc4')
    
    out = tf.contrib.layers.fully_connected(inputs=fc4, num_outputs=1, activation_fn=tf.nn.tanh, scope='out')

    return out

    
def r_net_dropout(state_input, action_input, f1=1, k1=5, f2=2, k2=3, n_fc3=8, n_fc4=4, n_fc5=4, d=15):
    
    action_input = tf.reshape(action_input, [-1,d,d,1])
    state_input = tf.reshape(state_input, [-1,d])
    
    conv1 = tf.contrib.layers.conv2d(inputs=action_input, num_outputs=f1, kernel_size=k1, stride=1, padding="SAME", activation_fn=tf.nn.relu, scope='conv1')
    
    conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=f2, kernel_size=k2, stride=1, padding="SAME", activation_fn=tf.nn.relu, scope='conv2')
    # flatten to vector
    conv2_flat = tf.reshape(conv2, [-1, f2*d*d]) 
    fc3 = tf.contrib.layers.fully_connected(inputs=conv2_flat, num_outputs=n_fc3, activation_fn=tf.nn.relu, scope='fc3') # batch_size x 64
    dropout3 = tf.contrib.layers.dropout(fc3, keep_prob=0.4, scope='dropout3')
    # for each sample in batch, concatenate conved action with state
    fc3_action = tf.concat([dropout3, state_input], 1)
    # one more fc layer
    fc4 = tf.contrib.layers.fully_connected(inputs=fc3_action, num_outputs=n_fc4, activation_fn=tf.nn.relu, scope='fc4')
    dropout4 = tf.contrib.layers.dropout(fc4, keep_prob=0.4, scope='dropout4')
    
    out = tf.contrib.layers.fully_connected(inputs=dropout4, num_outputs=1, activation_fn=tf.nn.tanh, scope='out')

    return out
