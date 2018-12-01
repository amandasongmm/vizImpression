from datetime import datetime
import pytz
import time
import os
import pandas as pd
import numpy as np
import pickle as Pickle
import cv2
import skimage.io
import skimage.transform
import tensorflow as tf

def data_augmentation(file):
    image = cv2.imread(os.path.join(image_path,file))
    horizontal_img = image.copy()
    horizontal_img = cv2.flip(horizontal_img,1)
    newname = os.path.splitext(file)[0]+"_flip.jpg"
    cv2.imwrite(os.path.join(image_path,newname),horizontal_img)
    return newname
 
def load_image( path ):
    try:
        img = skimage.io.imread( path ).astype( float )
    except:
        return None

    if img is None: return None
    if len(img.shape) < 2: return None
    if len(img.shape) == 4: return None
    if len(img.shape) == 2: img=np.tile(img[:,:,None], 3)
    if img.shape[2] == 4: img=img[:,:,:3]
    if img.shape[2] > 4: return None

    img /= 255.

    short_edge = min( img.shape[:2] )
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy+short_edge, xx:xx+short_edge]
    resized_img = skimage.transform.resize( crop_img, [224,224] , mode='constant')     #resize the image here
    return resized_img   


root_path = '/home/ghao/vizImpression'
image_path = os.path.join(root_path,'datasets/face_impression/images')
trainset_path = os.path.join(root_path,'datasets/face_impression/train.pickle')
weight_path = os.path.join(root_path,'trained_models/pretrained_weight/VGG/caffe_layers_value.pickle')
pretrained_model = None
model_path = os.path.join(root_path,'trained_models/VGG/')
saved_model_name = 'facemodel-'


#read data   
trainset = pd.read_pickle(trainset_path)
print ('Read from disk: trainset')

#CAM class
class Detector():
    def __init__(self, weight_file_path, n_labels):
        self.image_mean = [103.939, 116.779, 123.68]
        self.n_labels = n_labels

        with open(weight_file_path,'rb') as f:
            self.pretrained_weights = Pickle.load(f,encoding='iso-8859-1')

    def get_weight( self, layer_name):
        layer = self.pretrained_weights[layer_name]
        return layer[0]

    def get_bias( self, layer_name ):
        layer = self.pretrained_weights[layer_name]
        return layer[1]

    def get_conv_weight( self, name ):
        f = self.get_weight( name )
        return f.transpose(( 2,3,1,0 ))

    def conv_layer( self, bottom, name ):
        with tf.variable_scope(name) as scope:

            w = self.get_conv_weight(name)
            b = self.get_bias(name)

            conv_weights = tf.get_variable(
                    "W",
                    shape=w.shape,
                    initializer=tf.constant_initializer(w)
                    )
            conv_biases = tf.get_variable(
                    "b",
                    shape=b.shape,
                    initializer=tf.constant_initializer(b)
                    )

            conv = tf.nn.conv2d( bottom, conv_weights, [1,1,1,1], padding='SAME')
            bias = tf.nn.bias_add( conv, conv_biases )
            relu = tf.nn.relu( bias, name=name )

        return relu

    def new_conv_layer( self, bottom, filter_shape, name ):
        with tf.variable_scope( name ) as scope:
            w = tf.get_variable(
                    "W",
                    shape=filter_shape,
                    initializer=tf.random_normal_initializer(0., 0.01))
            b = tf.get_variable(
                    "b",
                    shape=filter_shape[-1],
                    initializer=tf.constant_initializer(0.))

            conv = tf.nn.conv2d( bottom, w, [1,1,1,1], padding='SAME')
            bias = tf.nn.bias_add(conv, b)

        return bias #relu

    def fc_layer(self, bottom, name, create=False):
        shape = bottom.get_shape().as_list()
        dim = np.prod( shape[1:] )
        x = tf.reshape(bottom, [-1, dim])

        cw = self.get_weight(name)
        b = self.get_bias(name)

        if name == "fc6":
            cw = cw.reshape((4096, 512, 7,7))
            cw = cw.transpose((2,3,1,0))
            cw = cw.reshape((25088,4096))
        else:
            cw = cw.transpose((1,0))

        with tf.variable_scope(name) as scope:
            cw = tf.get_variable(
                    "W",
                    shape=cw.shape,
                    initializer=tf.constant_initializer(cw))
            b = tf.get_variable(
                    "b",
                    shape=b.shape,
                    initializer=tf.constant_initializer(b))

            fc = tf.nn.bias_add( tf.matmul( x, cw ), b, name=scope)

        return fc

    def new_fc_layer( self, bottom, input_size, output_size, name ):
        shape = bottom.get_shape().to_list()
        dim = np.prod( shape[1:] )
        x = tf.reshape( bottom, [-1, dim])

        with tf.variable_scope(name) as scope:
            w = tf.get_variable(
                    "W",
                    shape=[input_size, output_size],
                    initializer=tf.random_normal_initializer(0., 0.01))
            b = tf.get_variable(
                    "b",
                    shape=[output_size],
                    initializer=tf.constant_initializer(0.))
            fc = tf.nn.bias_add( tf.matmul(x, w), b, name=scope)

        return fc

    def inference( self, rgb, train=False ):
        rgb *= 255.
        
        r, g, b = tf.split(rgb, num_or_size_splits=3, axis=3)
        bgr = tf.concat(
            [
                b-self.image_mean[0],
                g-self.image_mean[1],
                r-self.image_mean[2]
            ], axis=3)
        '''
        #OldTF
        r, g, b = tf.split(3, 3, rgb)
        bgr = tf.concat(3,
            [
                b-self.image_mean[0],
                g-self.image_mean[1],
                r-self.image_mean[2]
            ])
        '''

        relu1_1 = self.conv_layer( bgr, "conv1_1" )
        relu1_2 = self.conv_layer( relu1_1, "conv1_2" )

        pool1 = tf.nn.max_pool(relu1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                         padding='SAME', name='pool1')

        relu2_1 = self.conv_layer(pool1, "conv2_1")
        relu2_2 = self.conv_layer(relu2_1, "conv2_2")
        pool2 = tf.nn.max_pool(relu2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool2')

        relu3_1 = self.conv_layer( pool2, "conv3_1")
        relu3_2 = self.conv_layer( relu3_1, "conv3_2")
        relu3_3 = self.conv_layer( relu3_2, "conv3_3")
        pool3 = tf.nn.max_pool(relu3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool3')

        relu4_1 = self.conv_layer( pool3, "conv4_1")
        relu4_2 = self.conv_layer( relu4_1, "conv4_2")
        relu4_3 = self.conv_layer( relu4_2, "conv4_3")
        pool4 = tf.nn.max_pool(relu4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool4')

        relu5_1 = self.conv_layer( pool4, "conv5_1")
        relu5_2 = self.conv_layer( relu5_1, "conv5_2")
        relu5_3 = self.conv_layer( relu5_2, "conv5_3")

        conv6 = self.new_conv_layer( relu5_3, [3,3,512,1024], "conv6")
        gap = tf.reduce_mean( conv6, [1,2] )

        with tf.variable_scope("GAP"):
            gap_w = tf.get_variable(
                    "W",
                    shape=[1024, self.n_labels],
                    initializer=tf.random_normal_initializer(0., 0.01))

        output = tf.matmul( gap, gap_w)

        return pool1, pool2, pool3, pool4, relu5_3, conv6, gap, output

    def get_classmap(self, label, conv6):
        conv6_resized = tf.image.resize_bilinear( conv6, [224, 224] )
        with tf.variable_scope("GAP", reuse=True):
            label_w = tf.gather(tf.transpose(tf.get_variable("W")), label)
            label_w = tf.reshape( label_w, [-1, 1024, 1] ) # [batch_size, 1024, 1]

        conv6_resized = tf.reshape(conv6_resized, [-1, 224*224, 1024]) # [batch_size, 224*224, 1024]

        classmap = tf.matmul( conv6_resized, label_w )
        '''
        #OldTF
        classmap = tf.batch_matmul( conv6_resized, label_w )
        '''
        
        classmap = tf.reshape( classmap, [-1, 224,224] )
        return classmap



#Training 
n_epochs = 10
init_learning_rate = 0.0001
weight_decay_rate = 0.0005
momentum = 0.9
batch_size = 20

now = datetime.now(pytz.timezone('US/Eastern'))
seconds_since_epoch_start = time.mktime(now.timetuple())

graph = tf.Graph()
with graph.as_default():
    learning_rate = tf.placeholder( tf.float32, [])   #learning rate
    images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")       #image placeholder

    #Modify: placeholder's size
    labels_tf = tf.placeholder( tf.float32, [None,40], name='labels')                   #label placeholder

    detector = Detector(weight_path,40)

    p1,p2,p3,p4,conv5, conv6, gap, output = detector.inference(images_tf)          #return each conv
    
    #Modify: MSE loss function
    loss_tf = tf.losses.mean_squared_error(labels = labels_tf,predictions=output) 

    weights_only = filter(lambda x: x.name.endswith('W:0'), tf.trainable_variables())
    weight_decay = tf.reduce_sum(tf.stack([tf.nn.l2_loss(x) for x in weights_only])) * weight_decay_rate
    
    loss_tf += weight_decay                                                        #update
    saver = tf.train.Saver( max_to_keep=50 )

    optimizer = tf.train.MomentumOptimizer( learning_rate, momentum )
    grads_and_vars = optimizer.compute_gradients( loss_tf )
    grads_and_vars = map(lambda gv: (gv[0], gv[1]) if ('conv6' in gv[1].name or 'GAP' in gv[1].name) else (gv[0]*0.1, gv[1]), grads_and_vars)
    train_op = optimizer.apply_gradients( grads_and_vars )
    
with tf.Session(graph=graph) as sess:    
    tf.global_variables_initializer().run()
    '''
    #OldTF
    tf.initialize_all_variables().run()
    '''

    if pretrained_model:
        print ('Pretrained model loaded from ' + pretrained_model + ' (this overwrites the initial weights loaded to the model)')
        saver.restore(sess, pretrained_model)


    iterations = 0
    loss_list = []
    print ('Starting the training ...')
    for epoch in range(n_epochs):
        trainset.index = range(len(trainset))
        #Shuffle the index of all the trainset
        trainset = trainset.loc[np.random.permutation(len(trainset) )]
        
        for start, end in zip(
            range( 0, len(trainset)+batch_size, batch_size),
            range(batch_size, len(trainset)+batch_size, batch_size)):

            current_data = trainset[start:end]
            current_image_paths = current_data['image_path'].values    #return batch imagePaths with type of np array
            
            #Modify: image path
            current_images = np.array(list(map(lambda x: load_image(os.path.join(image_path,x)), current_image_paths)))

            good_index = np.array(list(map(lambda x: x is not None, current_images)))

            current_data = current_data[good_index]
            current_images = np.stack(current_images[good_index])

            
            # Obtaining the label of each image
            # transform it into a None*44 2d matrix
            current_labels = np.array(current_data['label'].values)            
            current_labels_deal = np.zeros((current_labels.shape[0],40))
            for index,row in enumerate(current_labels):
                current_labels_deal[index,:] = row
            #print(current_labels_deal.shape)
            #print(current_labels_deal)
            # Run tensorflow session to start train
            _, loss_val, output_val = sess.run(
                    [train_op, loss_tf, output],
                    feed_dict={
                        learning_rate: init_learning_rate,
                        images_tf: current_images,
                        labels_tf: current_labels_deal
                        })
            
            print("loss",loss_val)
            loss_list.append(loss_val)   #store the loss value

            iterations += 1            
            #Print out every 10 iterations
            if iterations % 10 == 0:
                print ("======================================")
                print ("Epoch", epoch + 1, "Iteration", iterations)
                print ("Processed", start, '/', len(trainset))
                print ("Training Loss:", np.mean(loss_list))
                print ("======================================")
                loss_list = []
        print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print ("producing model after epoch:{}".format(epoch+1))
        print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        saver.save( sess, os.path.join(model_path,saved_model_name), global_step=epoch)
        init_learning_rate *= 0.99
    
now = datetime.now(pytz.timezone('US/Eastern'))
seconds_since_epoch_end = time.mktime(now.timetuple())
print ('Processing took ' + str( np.around( (seconds_since_epoch_end - seconds_since_epoch_start)/60.0 , decimals=1) ) + ' minutes.')