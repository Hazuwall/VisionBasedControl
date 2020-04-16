import matplotlib.pyplot as plt
import simulation
import detection
import trajectory
import numpy as np
import tensorflow as tf

def depth_map_test():
    robot = simulation.Robot()
    robot.enable_synchronization()
    robot.reset()
    img, depth_map = robot.get_vision_feedback()
    plt.imshow(depth_map, "gray")
    plt.show()
    rect = detection.ObjectDetector().detect(img)
    d = trajectory.prepare_state_features(np.zeros((6)), img, depth_map, rect)[0]
    print(d)

def find_and_approach_test(n):
    robot = simulation.Robot()
    detector = detection.ObjectDetector()
    searcher = trajectory.Searcher(robot.joint_ranges[0,:])

    control_network = trajectory.create_control_network()
    weights_path = tf.train.latest_checkpoint("control_network")
    control_network.load_weights(weights_path)
    for i in range(n):
        robot.reset(is_dynamic=True, do_orientate=False)
        searcher.reset()
        terminal = False
        step = 0
        is_default = True
        while (not terminal) and (step < 35):
            state = trajectory.get_state(robot, detector)
            area = state[3]*state[4]
            if area > 0.2:
                terminal = True
            else:
                is_found = area > 0.001
                pos = trajectory.extract_pos(state)
                if is_default:
                    searcher.mark_coord(pos[0], is_found)
                if is_found:
                    action = control_network(np.expand_dims(state, axis=0)).numpy()
                    action = np.squeeze(action, axis=0)
                    next_pos = robot.clip_position(pos + action)
                    is_default = False
                else:
                    next_pos = robot.default_pos.copy()
                    next_pos[0] = searcher.get_coord(pos[0])
                    is_default = True
                success = robot.move(next_pos)
                if (not success) and is_default:
                    searcher.mark_coord(next_pos[0], False)
            step+=1

find_and_approach_test(5)
