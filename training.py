import tensorflow as tf
import numpy as np
import random
import simulation
import detection
import trajectory
from collections import deque

start_step = 0
episode = 0
display_step = 10
steps = 10000

batch_size = 70
memory_size = batch_size * 50

noise_theta = 0.02
noise_sigma = 0.025

weight_update_rate = 0.001

def limit_over(x, min, max):
    return np.maximum(min - x, 0) + np.maximum(x - max, 0)

def estimate_state_value(state, pos_t, joint_ranges):
    obj_area = state[3]*state[4]
    offset = np.sqrt(state[1]**2 + state[2]**2)
    depth = state[0]
    limit_penalty = 0.25 * np.mean(limit_over(pos_t, joint_ranges[:,0], joint_ranges[:,1]))
    value = obj_area + 0.5*(0.71 - offset) + 0.25*(1-depth) - limit_penalty
    terminal = 0
    if obj_area < 0.001:
        value += -1
        terminal = 1
    elif value > 0.2 + 0.3 + 0.2:
        value += 1
        terminal = 1
    return value, terminal, [limit_penalty]

def get_reward(state_value, next_state_value, terminal, episode_time, action):
    if episode_time > 15:
        return -0.5, 1, [0,0]
    else:
        rates = np.asarray([2,1,2,1,2,3])
        L1 = np.sum(abs(action)*rates)
        size_penalty = 0.5 * np.maximum(L1 - 1.4, 0)
        state_value_reward = next_state_value - state_value
        reward = state_value_reward + 0.03 - size_penalty
        return reward, terminal, [size_penalty, state_value_reward]

def update_model(old, new, rate):
    new_w = new.get_weights()
    w = old.get_weights()
    for i in range(len(new_w)):
        w[i] = w[i]*(1-rate) + new_w[i] * rate
    old.set_weights(w)

def compute_loss(actions, q, rewards, q_t):
    labels = rewards + 0.99 * q_t
    critic_cost = tf.reduce_mean(tf.keras.losses.MSE(labels, q))
    actor_cost = tf.reduce_mean(q * actions)
    return actor_cost, critic_cost

class ReplayMemory:
    def __init__(self, size):
        self.memory = deque(maxlen=size)
        
    def __len__(self):
        return len(self.memory)

    def append(self, entry):
        self.memory.append(entry)

    def sample(self, n):
        if n > len(self.memory): n = len(self.memory)
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
    def __init__(self, mu, theta, sigma, shape):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.shape = shape
        self.reset()
        
    def reset(self):
        self.x = np.ones(self.shape, dtype=np.float32) * self.mu
        
    def sample(self):  # dt = 1
        dx = self.theta * (self.mu - self.x) + self.sigma * np.random.normal(size=self.shape)
        self.x += dx
        return self.x

robot = simulation.Robot()
robot.enable_synchronization()
detector = detection.ObjectDetector()

actor = trajectory.create_control_network()
actor_target = trajectory.create_control_network()
update_model(actor_target, actor, 1)
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

critic = trajectory.create_qnetwork()
critic_target = trajectory.create_qnetwork()
update_model(critic_target, critic, 1)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
if start_step > 0:
    actor.load_weights("control_network\\checkpoint-" + str(start_step))
    critic.load_weights("q_network\\checkpoint-" + str(start_step))

memory = ReplayMemory(memory_size)
noise = OUNoise(0, noise_theta, noise_sigma, (6))
summary_writer = tf.summary.create_file_writer("logs")

@tf.function(experimental_relax_shapes=True)
def actor_train_step(states, step):
    with summary_writer.as_default():
        actions = actor(states, training=True)
        q = critic([actions, states], training=False)
        if step % display_step == 0:
            tf.summary.scalar('actor_q', tf.reduce_mean(q[:-5]), step)
        
        action_gradients = tf.concat(tf.gradients(q, actions), axis=0)
        vars = actor.trainable_variables
        unnormalized_gradients = tf.gradients(actions, vars, -action_gradients)
        count = tf.cast(tf.shape(actions)[0], dtype=tf.float32)
        gradients = list(map(lambda x: tf.math.divide(x, count), unnormalized_gradients))
        actor_optimizer.apply_gradients(zip(gradients, vars))

@tf.function(experimental_relax_shapes=True)
def critic_train_step(states, actions, rewards, next_states, terminals, step):
    terminals = tf.cast(terminals, dtype=tf.float32)
    terminals = tf.expand_dims(terminals, axis=1)
    rewards = tf.cast(rewards, dtype=tf.float32)
    rewards = tf.expand_dims(rewards, axis=1)
    with summary_writer.as_default():
        with tf.GradientTape() as tape:
            next_actions = actor_target(next_states, training=False)
            next_actions = tf.stop_gradient(next_actions)
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
            
# Training
print("Optimization Started!")
terminal = 1
total_reward = 0
episode_time = 0
for step in tf.range(start_step, start_step + steps + 1, dtype=tf.int64):
    # Инициализировать новую попытку
    if terminal == 1:
        with summary_writer.as_default():
            tf.summary.scalar('total_reward', total_reward, episode)
            tf.summary.scalar('episode_time', episode_time, episode)
        episode += 1
        episode_time = 0
        total_reward = 0
        noise.reset()
        while terminal == 1:
            robot.reset()
            next_state = trajectory.get_state(robot, detector)
            pos = trajectory.extract_pos(next_state)
            next_state_value, terminal, _ = estimate_state_value(next_state, pos, robot.joint_ranges)
    
    state = np.copy(next_state)
    state_value = next_state_value
    if step > 100:
        action = actor(np.expand_dims(state, axis=0), training=False).numpy()
        action = np.squeeze(action, axis=0)
    else:
        action = np.zeros((6), dtype=np.float32)
    action += noise.sample()
    pos = trajectory.extract_pos(state)
    robot.set_position(pos + action)
    episode_time += 1
    next_state = trajectory.get_state(robot, detector)
    next_state_value, terminal, state_metrics = estimate_state_value(next_state, pos + action, robot.joint_ranges)
    reward, terminal, reward_metrics = get_reward(state_value, next_state_value, terminal, episode_time, action)
    total_reward += reward
    with summary_writer.as_default():
        tf.summary.scalar('reward', reward, step)
        tf.summary.scalar('limit_penalty', state_metrics[0], step)
        tf.summary.scalar('size_penalty', reward_metrics[0], step)
        tf.summary.scalar('state_value_reward', reward_metrics[1], step)

    memory.append((state, action, reward, next_state, terminal))
    states, actions, rewards, next_states, terminals = memory.sample(batch_size)
    critic_train_step(states, actions, rewards, next_states, terminals, step)
    actor_train_step(states, step)
    update_model(actor_target, actor, weight_update_rate)
    update_model(critic_target, critic, weight_update_rate)

    summary_writer.flush()
    int_step = int(step)
    if (int_step % 1000) == 0:
        actor.save_weights("control_network\\checkpoint-" + str(int_step))
        critic.save_weights("q_network\\checkpoint-" + str(int_step))
print("Optimization Finished!")