import tensorflow as tf
import numpy as np

STATE_SIZE = 11 # размерность пространства состояний
ACTION_SIZE = 6 # размерность пространства действий

def create_qnetwork():
    """Создать нейросеть-критика.

    Returns
    -------
    tf.keras.Model
        Модель, которая принимает [действие, состояние] и возвращает числовую оценку
    """
    input1 = tf.keras.Input(shape=(ACTION_SIZE))
    input2 = tf.keras.Input(shape=(STATE_SIZE))
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
    return model

def create_control_network():
    """Создать нейросеть-актера.

    Returns
    -------
    tf.keras.Model
        Модель, которая принимает состояние и возвращает действие - изменение по координатам
    """

    input = tf.keras.Input(shape=(STATE_SIZE))
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

    output = tf.keras.layers.Dense(ACTION_SIZE, activation="tanh")(x) / 2.0
    model = tf.keras.Model(inputs=input, outputs=output)
    return model

def update_model(old, new, rate):
    """Обновить веса в нейронной сети

    Parameters
    ----------
    old : tf.keras.Model
        Модель, веса которой нужно обновить
    new : tf.keras.Model
        Модель, веса которой нужно использовать для обновления
    rate : float
        Экспоненциальный коэффициент сглаживания - скорость обновления
    """
    new_w = new.get_weights()
    w = old.get_weights()
    for i in range(len(new_w)):
        w[i] = w[i]*(1-rate) + new_w[i] * rate
    old.set_weights(w)

def prepare_state_features(pos, depth_map, obj_rect):
    """Подготовить вектор состояния

    Parameters
    ----------
    pos : ndarray
        Текущие обобщённые координаты
    depth_map : ndarray
        Карта глубины стереоизображения
    obj_rect : list
        Прямоугольник [x,y,width,height], обрамляющий объект на изображении

    Returns
    -------
    ndarray
        Вектор состояния
    """
    X = np.zeros((STATE_SIZE), dtype=np.float32)
    x,y,w,h = obj_rect

    if w != 0:
        width = depth_map.shape[0]
        height = depth_map.shape[1]

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
    """Получить состояние.

    Обёртка для определения текущего состояние с помощью обратной связи робота и детектора.

    Parameters
    ----------
    robot : Robot
        Управляемый робот
    detector : ObjectDetector
        Используемый детектор объекта

    Returns
    -------
    ndarray
        Вектор состояния
    """
    img, map = robot.get_vision_feedback()
    pos = robot.get_adometry_feedback()
    obj_loc = detector.detect(img)
    return prepare_state_features(pos, map, obj_loc)

def extract_pos(state):
    """Извлечь обобщённые координаты из вектора состояния

    Parameters
    ----------
    state : ndarray
        Вектор состояния

    Returns
    -------
    ndarray
        Обобщённые координаты
    """
    if len(state.shape) == 2:
        return state[:, 5:]
    else:
        return state[5:]