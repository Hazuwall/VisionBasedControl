import tensorflow as tf
import numpy as np
import random
from collections import deque
from environment import Robot
from detection import ObjectDetector
import network_utils

# Параметры

steps = 5000
start_step = 0
episode = 0
display_step = 10

batch_size = 70
memory_size = batch_size * 50

noise_theta = 0.02
noise_sigma = 0.025

weight_update_rate = 0.001


# Вычисление награды за действие

def limit_over(x, min, max):
    """Абсолютное значение, на сколько значение выходит за диапазон

    Parameters
    ----------
    x : ndarray
        Исследуемые значения
    min : ndarray
        Минимальные значения диапазона
    max : ndarray
        Максимальные значения диапазона

    Returns
    -------
    ndarray
        Размеры выхода за диапазон
    """
    return np.maximum(min - x, 0) + np.maximum(x - max, 0)

def estimate_state_value(state, pos_t, joint_ranges):
    """Оценить состояние

    Parameters
    ----------
    state : ndarray
        Вектор состояния
    pos_t : ndarray
        Целевые обобщённые координаты
    joint_ranges : ndarray
        Диапазоны координат

    Returns
    -------
    float
        Ценность состояния
    int
        1 - состояние заканчивает эпизод, 0 - нет
    dict
        Метрики
    """

    obj_area = state[3]*state[4]
    offset = np.sqrt(state[1]**2 + state[2]**2)
    depth = state[0]

    # Выход заданного перемещения за пределы рабочей зоны легко отбросить. Однако за него следует
    # отнимать очки, чтобы дать сети более быструю обратную связь о целесообразности действий.
    limit_penalty = 0.25 * np.mean(limit_over(pos_t, joint_ranges[:,0], joint_ranges[:,1]))

    value = obj_area + 0.5*(0.71 - offset) + 0.25*(1-depth) - limit_penalty
    terminal = 0
    if obj_area < 0.001:
        value += -1
        terminal = 1
    elif value > 0.2 + 0.3 + 0.2:
        value += 1
        terminal = 1
    return value, terminal, {"limit_penalty": limit_penalty}

def compute_reward(state_value, next_state_value, terminal, episode_time, action):
    """Вычислить награду

    Parameters
    ----------
    state_value : float
        Ценность состояния
    next_state_value : float
        Ценность следующего состояния
    terminal : int
        1 - состояние заканчивает эпизод, 0 - нет
    episode_time : int
        Длительность эпизода
    action : ndarray
        Действие - изменения по обобщённым координатам

    Returns
    -------
    float
        Награда за действие
    int
        1 - состояние заканчивает эпизод, 0 - нет
    list
        Метрики
    """

    if episode_time > 15:
        return -0.5, 1, [0,0]
    else:
        rates = np.asarray([2,1,2,1,2,3])
        L1 = np.sum(abs(action)*rates)
        # Размер перемещения может привести к потери объекта, поэтому вызывает штраф
        size_penalty = 0.5 * np.maximum(L1 - 1.4, 0)
        state_value_reward = next_state_value - state_value
        reward = state_value_reward + 0.03 - size_penalty
        return reward, terminal, { "size_penalty": size_penalty, "state_value_reward": state_value_reward }


# Фабрики для функций шагов обучения

def create_actor_train_step(actor, critic, actor_optimizer, summary_writer):
    @tf.function(experimental_relax_shapes=True)
    def actor_train_step(states, step):
        with summary_writer.as_default():
            actions = actor(states, training=True)
            q = critic([actions, states], training=False)
            if step % display_step == 0:
                tf.summary.scalar('actor_q', tf.reduce_mean(q[:-5]), step)
            
            # Вычисление производных по весам сети-актера,
            # используя частные производные Q по многомерному действию (изменениям обобщённых координат)
            action_gradients = tf.concat(tf.gradients(q, actions), axis=0)
            vars = actor.trainable_variables
            unnormalized_gradients = tf.gradients(actions, vars, -action_gradients)
            count = tf.cast(tf.shape(actions)[0], dtype=tf.float32)
            normalized_gradients = list(map(lambda x: tf.math.divide(x, count), unnormalized_gradients))
            actor_optimizer.apply_gradients(zip(normalized_gradients, vars))
    return actor_train_step

def create_critic_train_step(actor, actor_target, critic, critic_target, critic_optimizer, summary_writer):
    @tf.function(experimental_relax_shapes=True)
    def critic_train_step(states, actions, rewards, next_states, terminals, step):
        terminals = tf.cast(terminals, dtype=tf.float32)
        terminals = tf.expand_dims(terminals, axis=1)
        rewards = tf.cast(rewards, dtype=tf.float32)
        rewards = tf.expand_dims(rewards, axis=1)
        with summary_writer.as_default():
            with tf.GradientTape() as tape:
                # Следующее действие вычисляется с помощью копии актера, веса которого экспоненциально сглаживаются
                next_actions = actor_target(next_states, training=False)
                next_actions = tf.stop_gradient(next_actions)
                # Оценка Q следующего действия также вычисляется с помощью копии критика
                q_t = critic_target([next_actions, next_states], training=False)
                q_t *= 1 - terminals
                q_t = tf.stop_gradient(q_t)
                q = critic([actions, states], training=True)
                labels = rewards + 0.99 * q_t
                cost = tf.reduce_mean(tf.keras.losses.MSE(labels, q))
                
                if step % display_step == 0:
                    tf.summary.scalar('critic_cost', cost, step)

                vars = critic.trainable_variables
                gradients = tape.gradient(cost, vars)
                critic_optimizer.apply_gradients(zip(gradients, vars))
    return critic_train_step


# Дополнительные сущности для улучшения обучения

class ReplayMemory:
    """Буфер для воспроизведения полученного ранее опыта
    """

    def __init__(self, size):
        self.memory = deque(maxlen=size)
        
    def __len__(self):
        return len(self.memory)

    def append(self, entry):
        self.memory.append(entry)

    def sample(self, n):
        if n > len(self.memory):
            n = len(self.memory)
        entries = random.sample(self.memory, n)
        batch = []
        entry_len = len(entries[0])
        for i in range(entry_len):
            temp = []
            for j in range(n):
                temp.append(entries[j][i])
            batch.append(np.stack(temp, axis=0))
        return batch

class OUNoise:
    """Шум Орнштейна-Уленбека
    """

    def __init__(self, mu, theta, sigma, shape):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.shape = shape
        self.reset()
        
    def reset(self):
        self.x = np.ones(self.shape, dtype=np.float32) * self.mu
        
    def sample(self):
        dx = self.theta * (self.mu - self.x) + self.sigma * np.random.normal(size=self.shape)
        self.x += dx
        return self.x


# Инициализация объектов

robot = Robot()
robot.enable_synchronization() # обучение проходит на статической сцене в режиме синхронизации
detector = ObjectDetector()

actor = network_utils.create_control_network()
actor_target = network_utils.create_control_network()
network_utils.update_model(actor_target, actor, 1)
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

critic = network_utils.create_qnetwork()
critic_target = network_utils.create_qnetwork()
network_utils.update_model(critic_target, critic, 1)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

summary_writer = tf.summary.create_file_writer("logs")
actor_train_step = create_actor_train_step(actor, critic, actor_optimizer, summary_writer)
critic_train_step = create_critic_train_step(actor, actor_target, critic, critic_target, critic_optimizer, summary_writer)

memory = ReplayMemory(memory_size)
noise = OUNoise(0, noise_theta, noise_sigma, (6))


# Обучение

print("Optimization Started!")
terminal = 1
total_reward = 0
episode_time = 0
for step in tf.range(start_step, start_step + steps + 1, dtype=tf.int64):
    # Инициализация новой попытки
    if terminal == 1:
        with summary_writer.as_default():
            tf.summary.scalar('total_reward', total_reward, episode)
            tf.summary.scalar('episode_time', episode_time, episode)
        episode += 1
        episode_time = 0
        total_reward = 0
        noise.reset()
        while terminal == 1:
            # После сброса цель должна находиться в зоне видимости объекта, 
            # данный цикл защищает от возможных исключений
            robot.reset(is_dynamic=False, do_orientate=True)
            next_state = network_utils.get_state(robot, detector)
            pos = network_utils.extract_pos(next_state)
            next_state_value, terminal, _ = estimate_state_value(next_state, pos, robot.joint_ranges)
    
    state, state_value = next_state, next_state_value
    episode_time += 1

    # Действия на первых шагах определяются случайным шумом, чтобы сконцентрировать на расширении
    # опыта и подтолкнуть к использованию небольших перемещений, так как цель изначально в зоне видимости
    if step > 100:
        action = actor(np.expand_dims(state, axis=0), training=False).numpy()
        action = np.squeeze(action, axis=0)
    else:
        action = np.zeros((6), dtype=np.float32)
    action += noise.sample()
    pos = network_utils.extract_pos(state)
    robot.set_position(pos + action)

    # Определение следующих за действием состояния и награды, сохранение набора в память
    next_state = network_utils.get_state(robot, detector)
    next_state_value, terminal, state_metrics = estimate_state_value(next_state, pos + action, robot.joint_ranges)
    reward, terminal, reward_metrics = compute_reward(state_value, next_state_value, terminal, episode_time, action)
    total_reward += reward
    with summary_writer.as_default():
        tf.summary.scalar('reward', reward, step)
        for key in state_metrics:
            tf.summary.scalar(key, state_metrics[key], step)
        for key in reward_metrics:
            tf.summary.scalar(key, reward_metrics[key], step)
    memory.append((state, action, reward, next_state, terminal))

    # Обновление весов на основе полученного опыта
    states, actions, rewards, next_states, terminals = memory.sample(batch_size)
    critic_train_step(states, actions, rewards, next_states, terminals, step)
    actor_train_step(states, step)
    network_utils.update_model(actor_target, actor, weight_update_rate)
    network_utils.update_model(critic_target, critic, weight_update_rate)

    summary_writer.flush()
    int_step = int(step)
    if (int_step % 1000) == 0:
        actor.save_weights("control_network\\checkpoint-" + str(int_step))
        critic.save_weights("q_network\\checkpoint-" + str(int_step))
actor.save("model.h5")
print("Optimization Finished!")