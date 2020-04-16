import sim
import numpy as np
import cv2 as cv
import random
import time

stereo_camera_name = "StereoCamera_"
joint_name = "LBR4p_joint"
scene_controller_name = "SceneController"
joint_range_func_name = "getJointRange"
reset_func_name = "resetItems"
joint_count=6

class Robot():
    def __init__(self):
        sim.simxFinish(-1) # close all opened connections
        self.client=sim.simxStart('127.0.0.1',19999,True,True,5000,5) # connect
        self.is_connected = self.client != -1
        self.synchronous = False
        self.cameras = []
        self.joints = []
        self.joint_ranges = np.zeros((joint_count,2), dtype=np.float32)
        self.default_pos = np.asarray([0, 0, 2.97, 2.62, 1.57, 0])
        self.scene_controller = 0
        self.stereo_matcher = cv.StereoBM_create(numDisparities=48, blockSize=11)

        if self.is_connected:
            _, self.scene_controller = sim.simxGetObjectHandle(self.client, scene_controller_name, sim.simx_opmode_blocking)
            for i in range(1,3):
                _, id = sim.simxGetObjectHandle(self.client, stereo_camera_name + str(i), sim.simx_opmode_blocking)
                self.cameras.append(id)
            for i in range(1,joint_count+1):
                _, id = sim.simxGetObjectHandle(self.client, joint_name + str(i), sim.simx_opmode_blocking)
                self.joints.append(id)
                _, _, min_max, _, _ = sim.simxCallScriptFunction(self.client, scene_controller_name, sim.sim_scripttype_childscript, joint_range_func_name,
                    [id],[],[],bytearray(), sim.simx_opmode_blocking)
                for j in range(2):
                    self.joint_ranges[i-1,j] = min_max[j]
            self.reset()
        else:
            print('Failed connecting to remote API server')

    def __del__(self):
        if self.synchronous:
            self.disable_synchronization()
        sim.simxFinish(self.client)
    
    def get_vision_feedback(self):
        # stereo images
        imgs = []
        for i in range(2):
            _, res, img = sim.simxGetVisionSensorImage(self.client, self.cameras[i], False, sim.simx_opmode_blocking)
            img = np.asarray(img, dtype=np.uint8)
            img = np.reshape(img, (res[0],res[1],3))
            img = np.flip(img, axis=0)
            imgs.append(img)
        
        # depth map
        left = cv.cvtColor(imgs[1], cv.COLOR_RGB2GRAY)
        right = cv.cvtColor(imgs[0], cv.COLOR_RGB2GRAY)
        depth_map = self.stereo_matcher.compute(left,right)
        return imgs[1], depth_map/752

    def get_adometry_feedback(self):
        pos = np.zeros((joint_count))
        for i in range(joint_count):
            _, pos[i] = sim.simxGetJointPosition(self.client, self.joints[i], sim.simx_opmode_blocking)
        return self.clip_position(pos)

    def clip_position(self, coords):
        return np.clip(coords, self.joint_ranges[:,0], self.joint_ranges[:,1])

    def set_target_position(self, coords):
        sim.simxPauseCommunication(self.client, 1)
        for i in range(joint_count):
            sim.simxSetJointTargetPosition(self.client, self.joints[i], coords[i], sim.simx_opmode_oneshot)
        sim.simxPauseCommunication(self.client, 0)

    def set_position(self, coords):
        sim.simxPauseCommunication(self.client, 1)
        for i in range(joint_count):
            sim.simxSetJointPosition(self.client, self.joints[i], coords[i], sim.simx_opmode_oneshot)
        sim.simxPauseCommunication(self.client, 0)
        if self.synchronous:
            self._step_simulation()

    def move(self, coords_t):
        if self.synchronous:
            raise ValueError()
        self.set_target_position(coords_t)
        last_e = 1000
        while True:
            coords = self.get_adometry_feedback()
            e = np.mean(np.abs(coords - coords_t))
            if e < 0.01:
                return True
            if last_e - e < 0.001:
                return False
            last_e = e
            time.sleep(0.2)

    def reset(self, is_dynamic=False, do_orientate=True):
        _, _, phi, _, _ = sim.simxCallScriptFunction(self.client, scene_controller_name, sim.sim_scripttype_childscript, reset_func_name,
                                    [],[],[],bytearray(), sim.simx_opmode_blocking)
        pos = self.default_pos.copy()
        if do_orientate:
            offset = random.random()*0.2 - 0.1
            pos[0] = phi[0] + offset
        else:
            pos[0] = random.random()*(self.joint_ranges[0,1]-self.joint_ranges[0,0]) + self.joint_ranges[0,0]
        
        if is_dynamic:
            self.move(pos)
        else:
            self.set_position(pos)

    def enable_synchronization(self):
        sim.simxSynchronous(self.client,True)
        self.synchronous = True

    def disable_synchronization(self):
        sim.simxSynchronous(self.client,False)
        self.synchronous =False

    def _step_simulation(self):
        sim.simxSynchronousTrigger(self.client)
        sim.simxGetPingTime(self.client)