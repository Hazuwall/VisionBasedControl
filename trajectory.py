import tensorflow as tf
import numpy as np
import os
from collections import deque

state_size = 11
action_size = 6
search_res = 20

def create_qnetwork():
    input1 = tf.keras.Input(shape=(action_size))
    input2 = tf.keras.Input(shape=(state_size))
    inputs = [input1, input2]
    x = tf.concat(inputs, axis=1)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    x = tf.keras.layers.Dense(64, activation="sigmoid")(x)
    output = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    #model.summary()
    return model

def create_control_network():
    input = tf.keras.Input(shape=(state_size))
    x = tf.concat(input, axis=1)
    x = tf.keras.layers.Dense(64)(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("sigmoid")(x)
    
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    output = tf.keras.layers.Dense(action_size, activation="tanh")(x) / 2.0
    model = tf.keras.Model(inputs=input, outputs=output)
    #model.summary()
    return model

def prepare_state_features(pos, img, depth_map, obj_rect):
    X = np.zeros((state_size), dtype=np.float32)
    x,y,w,h = obj_rect

    if w != 0:
        width = img.shape[0]
        height = img.shape[1]

        X[0] = np.mean(depth_map[x:x+w, y:y+h])
        if X[0] < 0:
            X[0] = 0
        X[1] = (x + w - width) / width / 2
        X[2] = (y + h - height) / height / 2
        X[3] = w / width
        X[4] = h / height
    X[5:] = pos
    return X

def get_state(robot, detector):
    img, map = robot.get_vision_feedback()
    pos = robot.get_adometry_feedback()
    obj_loc = detector.detect(img)
    return prepare_state_features(pos, img, map, obj_loc)

def extract_pos(state):
    if len(state.shape) == 2:
        return state[:, 5:]
    else:
        return state[5:]

class Searcher():
    def __init__(self, bounds):
        self.bounds = bounds
        self.res = search_res
        self.map = np.zeros((self.res))
    
    def get_coord(self, current_coord):
        indices = np.where(self.map == 1)[0]
        if len(indices) > 0:
            return self._get_coord(indices[0])
        
        indices = np.where(self.map == 0)[0]
        if len(indices) == 0:
            return current_coord
        current_index = self._get_index(current_coord)
        distance = np.abs(indices - current_index)
        nearest_index = indices[np.argmin(distance)]
        return self._get_coord(nearest_index)
            
    def mark_coord(self, coord, is_found):
        index = self._get_index(coord)
        self.map[index] = 1 if is_found else -1

    def reset(self):
        self.map = np.zeros((self.res))

    def _get_index(self, coord):
        index = self.res * (coord - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        return np.clip(int(index), 0, self.res-1)

    def _get_coord(self, index):
        return (self.bounds[1] - self.bounds[0]) * index / self.res + self.bounds[0]