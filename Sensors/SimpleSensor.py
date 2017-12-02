# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 09:56:32 2017

@author: nicu
"""

import argparse
import os
import glob
import math
import random
import numpy as np
import cv2
import tensorflow as tf
#import matplotlib.pyplot as plt

FLAGS = None

"""
Coordinates are screen-like, i.e. x=0, y=0 is top, left corner
"""

"""
This is a simple, deterministic sensor, which can detect
the presence of the object based on euclidian distance
and returns 1 if the object was detected or 0 otherwise
Example: PIR (passive infrared) sensors, ~$10, ceiling mounted,
in 5 m heigth, with a nominal range of 10 m (radius)
"""
class Sensor():
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    def detect(self, x, y):
        dx = x - self.x
        dy = y - self.y
        r = math.sqrt(dx * dx + dy * dy)
        return 1 if r <= self.r else 0

"""
A room has 2 dimensions and a list of sensors
"""
class Room():

    MAXEXT = 800
    MINEXT = 400

    def __init__(self, x, y, s):
        self.x = x
        self.y = y
        self.sensors = s
        mi, ma = (x, y) if x > y else (y, x)
        if ma > Room.MAXEXT:
            self.scale = Room.MAXEXT / ma
        elif mi < Room.MINEXT:
            self.scale = Room.MINEXT / mi
        else:
            self.scale = 1.

    def in_range(self, x, y):
        return 0 <= x <= self.x and 0 <= y <= self.y

    def detect(self, x, y):
        detects = []
        for s in self.sensors:
            detects.append(s.detect(x, y))
        return detects

    def img_coord(self, x, y):
        kx = int(x * self.scale)
        ky = int(y * self.scale)
        return kx, ky

    def draw(self):
        kx, ky = self.img_coord(self.x, self.y)
        img = np.zeros((ky, kx, 3), dtype=np.uint8)
        for s in self.sensors:
            cx, cy = self.img_coord(s.x, s.y)
            cv2.circle(img, (cx, cy), 3, (100, 100, 100), -1)
        return img

    def draw_detects(self, img, det, ps):
        for s, a in zip(self.sensors, det):
            if a:
                cx, cy = self.img_coord(s.x, s.y)
                cv2.circle(img, (cx, cy), 3, (10, 255, 20), -1)
        for p, _ in ps:
            cx, cy = self.img_coord(p.x, p.y)
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

"""
A person will appear at one random side of the given room
and will go to another random side of that room with a random speed
"""
class Person():

    SPEED = 1
    SVAR = 0.15
    SMIN = 0.2

    def __init__(self, room):
        # Select a random start and end side
        se = random.sample(('N', 'S', 'E', 'W'), 2)
        sx, sy = self.randpos(se[0], room)
        self.x = sx
        self.y = sy
        ex, ey = self.randpos(se[1], room)
        dx = ex - sx
        dy = ey - sy
        speed = 0
        while speed < Person.SMIN:
            speed = random.gauss(Person.SPEED, Person.SVAR)
        d = math.sqrt(dx * dx + dy * dy)
        t = d / speed
        self.vx = dx / t
        self.vy = dy / t

    def step(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt

    def randpos(self, side, room):
        if side == 'N':
            y = 0
            x = random.uniform(0, room.x)
        elif side == 'S':
            y = room.y
            x = random.uniform(0, room.x)
        elif side == 'E':
            x = room.x
            y = random.uniform(0, room.y)
        else:
            x = 0
            y = random.uniform(0, room.y)
        return x, y

"""
Episodes with only one person in a room
"""
def episode(room, dt):
    person = Person(room)
    dtms = int(dt * 1000)
    #print('Start at', person.x, person.y)
    rimg = room.draw()
    while room.in_range(person.x, person.y):
        cimg = rimg.copy()
        det = room.detect(person.x, person.y)
        room.draw_detects(cimg, det, [person])
        cv2.imshow('room', cimg)
        if cv2.waitKey(dtms) & 0xFF == ord('q'):
            break
        person.step(dt)

def rand_room(x, y, n, r):
    sensors = []
    for _ in range(n):
        s = Sensor(random.uniform(0, x), random.uniform(0, y), r)
        sensors.append(s)
    room = Room(x, y, sensors)
    return room

def grid_room(x, y, nx, ny, r):
    sensors = []
    dx = x / nx
    dy = y / ny
    for ix in range(nx):
        sx = dx / 2 + dx * ix
        for iy in range(ny):
            sy = dy / 2 + dy * iy
            s = Sensor(sx, sy, r)
            sensors.append(s)
    room = Room(x, y, sensors)
    return room

def comb_detect(det1, det2):
    l1 = len(det1)
    l2 = len(det2)
    if l1 > l2:
        return comb_detect_eq(det1[:l2], det2) + det1[l2:]
    elif l2 > l1:
        return comb_detect_eq(det1, det2[:l1]) + det2[l1:]
    else:
        return comb_detect_eq(det1, det2)

def comb_detect_eq(det1, det2):
    det = []
    for a, b in zip(det1, det2):
        det.append(max(a, b))
    return det

"""
Experiment with many persons traversing a square
- room: simulation room (or square)
- sen_x: number of sensors in the grid in horizontal direction
- sen_y: number of sensors in the grid in vertical direction
- pers_freq: frequency of a new person (per second)
- steps: similation steps
- maxtp: max traced persons
- history: number of kept input sensors frames
- dt: simulation time step
- draw: show the animation
- timescale: when drawing, how fast to simulate (1: real time, < 1: faster)
"""
def square(room, sen_x, sen_y, pers_freq, steps=1000, maxtp=32, history=16, draw=False, dt=0.04, timescale=1.):
    # This "time" must not be multiplied by the timescale
    # due to how the person generation works
    ptime = 1. / pers_freq
    # This is the simulation step time
    persons = []
    if draw:
        lp = 0
        rimg = room.draw()
        dtms = int(dt * 1000 * timescale)
    # Tensors to collect data: person mask (valid persons), sensor reading
    # and ground truth, which keeps the position (x, y) per tracked person
    M = np.zeros((steps, maxtp), dtype=np.float32)
    X = np.zeros((steps, sen_x, sen_y, history), dtype=np.float32)
    Y = np.zeros((steps, maxtp, 2), dtype=np.float32)
    pmask = np.zeros(maxtp, dtype=np.float32)
    hX = []
    for _ in range(history):
        deta = np.zeros((sen_x, sen_y), dtype=np.float32)
        hX.append(deta)
    rest = np.random.poisson(lam=ptime)
    for i in range(steps):
        if i % 200 == 0:
            print('step', i)
        rest -= dt
        if rest < 0:
            p = Person(room)
            # Find a place for the new person
            free = None
            k = 0
            while k < maxtp:
                if pmask[k] == 0.:
                    pmask[k] = 1.
                    free = k
                    break
                else:
                    k += 1
            persons.append((p, free))
            rest += np.random.poisson(lam=ptime)
        det = []
        for p, k in persons:
            # Detect every person
            det = comb_detect(det, room.detect(p.x, p.y))
            if k is not None:
                # Ground truth (this one is traced)
                Y[i, k, 0] = p.x
                Y[i, k, 1] = p.y
        if det == []:
            deta = np.zeros((sen_x, sen_y), dtype=np.float32)
        else:
            deta = np.array(det, dtype=np.float32)
            deta = np.reshape(deta, (sen_x, sen_y))
        hX.append(deta)
        hX = hX[1:]
        X[i, :, :, :] = np.stack(hX, axis=2)
        M[i, :] = pmask
        if draw:
            cimg = rimg.copy()
            room.draw_detects(cimg, det, persons)
            cv2.imshow('room', cimg)
            if cv2.waitKey(dtms) & 0xFF == ord('q'):
                break
        rps = []
        for p, k in persons:
            p.step(dt)
            if room.in_range(p.x, p.y):
                rps.append((p, k))
            elif k is not None:
                # Delete from mask if traced
                pmask[k] = 0.
        persons = rps
        if draw and len(persons) != lp:
            lp = len(persons)
            print(lp, 'persons')
    return X, M, Y

"""
Next functions deal with coding, writing and decoding TFRecord files
"""
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_tfrecords(X, M, Y, name, i):
    """Writes to tfrecords"""
    samples, sen_x, sen_y, hist = X.shape
    maxtp = M.shape[1]

    filename = os.path.join(FLAGS.directory, name + '_{:03d}.tfrecords'.format(i))
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(samples):
        X_raw = X[index].tostring()
        M_raw = M[index].tostring()
        Y_raw = Y[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'sen_x': _int64_feature(sen_x),
            'sen_y': _int64_feature(sen_y),
            'hist': _int64_feature(hist),
            'maxtp': _int64_feature(maxtp),
            'M_raw': _bytes_feature(M_raw),
            'Y_raw': _bytes_feature(Y_raw),
            'X_raw': _bytes_feature(X_raw)}))
        writer.write(example.SerializeToString())
    writer.close()

# If we want only to count the persons we do not need the positions (Y)
def decode_tfrecord(serialized):
    features = tf.parse_single_example(serialized,
                       features={
                               'M_raw': tf.FixedLenFeature([], tf.string),
                               'X_raw': tf.FixedLenFeature([], tf.string),
                       })
    M = tf.decode_raw(features['M_raw'], tf.float32)
    X = tf.decode_raw(features['X_raw'], tf.float32)
    return M, X

"""
Train a CNN for our square people problem - only count
"""
def train_cnn(maxtp, sen_x, sen_y, hist, kp=0.5, lr=0.001, batch_sz=50, shuffle_sz=25000):
    tf.reset_default_graph()

    # Prepare 2 datasets, train & dev
    train_files = glob.glob(os.path.join(FLAGS.directory, 'train_*.tfrecords'))
    train_ds = tf.contrib.data.TFRecordDataset(train_files).map(decode_tfrecord)
    train_ds = train_ds.shuffle(buffer_size=shuffle_sz).batch(batch_sz).repeat()

    dev_files = glob.glob(os.path.join(FLAGS.directory, 'dev_*.tfrecords'))
    dev_ds = tf.contrib.data.TFRecordDataset(dev_files).map(decode_tfrecord).batch(batch_sz)
    # Make a compatible feedable iterator for generic use
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.contrib.data.Iterator.from_string_handle(
            handle, train_ds.output_types, train_ds.output_shapes)
    # Create 2 iterators for train and dev
    train_iter = train_ds.make_initializable_iterator()
    dev_iter = dev_ds.make_initializable_iterator()

    # Here we build our graph
    # This is the input of our network
    with tf.name_scope('input'):
        mask, senflat = iterator.get_next()
        mask.set_shape([None, maxtp])
        senflat.set_shape([None, sen_x * sen_y * hist])
        sensors = tf.reshape(senflat, [-1, sen_x, sen_y, hist])

    is_training=tf.placeholder(dtype=tf.bool, shape=[])

    with tf.name_scope('cnn'):
        # Floyd run 3
        ## Convolution layers:
        #co1 = tf.contrib.layers.conv2d(sensors, 8, 3)
        #mp1 = tf.contrib.layers.max_pool2d(co1, 3, stride=3)
        #co2 = tf.contrib.layers.conv2d(mp1, 16, 3)
        #mp2 = tf.contrib.layers.max_pool2d(co2, 2, stride=2)
        ## FC layers
        #fl = tf.contrib.layers.flatten(mp2)
        #fc1 = tf.contrib.layers.fully_connected(fl, 32)

        # Floyd run 4
        # Convolution layers:
        co1 = tf.contrib.layers.conv2d(sensors, 16, 3)
        mp1 = tf.contrib.layers.max_pool2d(co1, 3, stride=3)
        co2 = tf.contrib.layers.conv2d(mp1, 16, 3)
        mp2 = tf.contrib.layers.max_pool2d(co2, 2, stride=2)
        # FC layers
        fl = tf.contrib.layers.flatten(mp2)
        fc1 = tf.contrib.layers.fully_connected(fl, 32)

        # Dropout
        dr1 = tf.contrib.layers.dropout(fc1, keep_prob=kp, is_training=is_training)
        # Final layer
        fcr = tf.contrib.layers.fully_connected(dr1, maxtp+1, activation_fn=None)

    with tf.name_scope('loss'):
        # Ground truth is the sum of the mask (number of tracked people)
        y = tf.reduce_sum(mask, axis=1)
        yoh = tf.contrib.layers.one_hot_encoding(tf.cast(y, tf.int32), maxtp+1)
        # Prediction
        topi = tf.cast(tf.argmax(tf.nn.softmax(fcr), axis=1), tf.float32)
        # Weight with squared class distance (+1: for same class, just normal classification cost)
        ws = tf.div(tf.losses.mean_squared_error(y, topi, reduction=tf.losses.Reduction.NONE) + 1, y + 1)
        loss = tf.losses.softmax_cross_entropy(yoh, fcr, ws)
        #print(y.shape, yoh.shape, topi.shape, ws.shape)

        # Summaries:
        tf.summary.scalar('loss', loss)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.save + '/summ_train', tf.get_default_graph())
        test_writer = tf.summary.FileWriter(FLAGS.save + '/summ_test')

    # Optimizer, init
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    init_op = tf.global_variables_initializer()

    # Saver
    vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
    saver = tf.train.Saver(vars, pad_step_number=True)

    with tf.Session() as sess:
        train_handle = sess.run(train_iter.string_handle())
        dev_handle = sess.run(dev_iter.string_handle())

        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Initializer for input iteration
        sess.run(train_iter.initializer)

        ## Test:
        #X = sess.run(sensors, feed_dict={handle: train_handle})
        #print(X.shape)
        vls = []
        tls = []

        cost_train = 0
        batch_train = 0
        for step in range(FLAGS.train_steps):
            #print('Step', step)
            if step > 0 and step % FLAGS.val_steps == 0:
                saver.save(sess=sess, global_step=step, save_path=FLAGS.save + '/checkpoint')
                cost_train = cost_train / batch_train
                tls.append(cost_train)
                cost_train = 0
                batch_train = 0

                sess.run(dev_iter.initializer)
                loss_dev = 0
                sqrsum   = 0
                accurate = 0
                batch_dev = 0
                examp = False
                while True:
                    try:
                        # This will take input from dev dataset
                        pred, truth, vloss, summary = sess.run([fcr, yoh, loss, merged], feed_dict={handle: dev_handle, is_training: False})
                        # Run the validation here
                        yi = np.argmax(truth, axis=1)
                        pi = np.argmax(pred, axis=1)
                        ok = (pi == yi)
                        accurate += np.sum(ok)
                        df = pi - yi
                        sqrsum += np.dot(df, df)
                        if not examp and np.random.rand() < 0.005:
                            print('True =', yi)
                            print('Pred =', pi)
                            examp = True
                        n = pred.shape[0]
                        batch_dev += n
                        loss_dev += vloss
                        test_writer.add_summary(summary, step)
                    except tf.errors.OutOfRangeError:
                        break
                loss_dev = loss_dev / batch_dev
                acc = accurate / batch_dev
                std = math.sqrt(sqrsum / batch_dev)
                print('Step', step, ': dev error =', loss_dev, 'accuracy:', acc, 'std err:', std)
                vls.append(loss_dev)

            _, pred, tl, summary = sess.run([train_op, fcr, loss, merged], feed_dict={handle: train_handle, is_training: True})
            train_writer.add_summary(summary, step)
            n = pred.shape[0]
            batch_train += n
            cost_train += tl

        coord.request_stop()
        coord.join(threads)

#    plt.plot(tls, label='Training')
#    plt.plot(vls, label='Validation')
#    plt.title('Train/Validation Loss')
#    plt.legend()
#    plt.show()

def main(unused_args):
#    #room = rand_room(100, 150, 100, 15)
    # 100mx50m area with 72 sensors (grid)
    width = 100
    height = 50
    sen_x = 12
    sen_y = 6
    sen_r = 10
    room = grid_room(width, height, sen_x, sen_y, sen_r)
#    for i in range(10):
#        print('Episode', i)
#        episode(room)
#    X, M, Y = square(room, sen_x, sen_y, draw=True, pers_freq=0.1, steps=1000, maxtp=32,
#                     history=8, dt=1, timescale=0.1)
    hlen = 12
    maxtp = 32
    steps = 1000
    fret = 1
    # Different persons frequencies, 1s/frame, training/dev/test data
    itr = 0
    idv = 0
    ite = 0
    for pf in [1/3, 1/4, 1/5, 1/6, 1/7, 1/8, 1/9, 1/10, 1/11, 1/12]:
        print('Freq:', pf)
        for _ in range(100):
            X, M, Y = square(room, sen_x, sen_y, pers_freq=pf, steps=steps, maxtp=maxtp,
                             history=hlen, dt=fret)
            write_tfrecords(X, M, Y, 'train', itr)
            itr += 1
        for _ in range(10):
            X, M, Y = square(room, sen_x, sen_y, pers_freq=pf, steps=steps, maxtp=maxtp,
                             history=hlen, dt=fret)
            write_tfrecords(X, M, Y, 'dev', idv)
            idv += 1
        for _ in range(10):
            X, M, Y = square(room, sen_x, sen_y, pers_freq=pf, steps=steps, maxtp=maxtp,
                             history=hlen, dt=fret)
            write_tfrecords(X, M, Y, 'test', ite)
            ite += 1
    print('X flat:', sen_x * sen_y * hlen)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory',
        type=str,
        default='../data/sensors',
        help='Directory to write the train/test data files'
    )
    parser.add_argument(
        '--save',
        type=str,
        default='../output/sensors/local',
        help='Directory to save the training result files'
    )
    parser.add_argument(
        '--train_steps',
        type=int,
        default=1000,
        help="""\
        Number of training steps
        set.\
        """
    )
    parser.add_argument(
        '--val_steps',
        type=int,
        default=200,
        help="""\
        Number of training steps after which a validation occurs
        set.\
        """
    )
    FLAGS, unparsed = parser.parse_known_args()
    #main(unparsed)
    train_cnn(maxtp=32, sen_x=12, sen_y=6, hist=12, kp=0.5, lr=1E-5)