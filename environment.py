import sim
import numpy as np
import cv2 as cv
import random
import time

STEREO_CAMERA_NAME = "StereoCamera_"
JOINT_NAME = "LBR4p_joint"
SCENE_CONTROLLER_NAME = "SceneController"

# функция возвращает минимальную и максимальную координаты сочленения
JOINT_RANGE_FUNC_NAME = "getJointRange"

# функция случайно расставляет объекты и возвращающая угол сектора целевого объекта
RESET_FUNC_NAME = "resetItems" 

# количество управляемых кинематических пар
JOINT_COUNT=6

class Robot():
    """Осуществляет взаимодействие с манипулятором и сценой

    Attributes
    -------
    client : int
        Id клиента, подключенного к среде CoppeliaSim
    is_connected : bool
        Подключен ли клиент к среде
    synchronous : bool
        Включен ли режим синхронизации
    cameras : list
        Id камер, образующих стереокамеру
    joints : list
        Id сочленений
    joint_ranges : ndarray
        Крайние обобщённые координаты
    default_pos : ndarray
        Координаты начального положения
    scene_controller : int
        Id объекта, управляющего окружением робота
    stereo_matcher : cv2.StereoBM
        Объект, вычисляющий карту глубины

    Methods
    -------
    get_vision_feedback()
        Получить визуальную обратную связь
    get_adometry_feedback()
        Получить обратную связь по положению
    clip_position(coords)
        Привести координаты к допустимому диапазону
    set_target_position(coords)
        Установить целевое положение (для динамической сцены)
    set_position(coords)
        Установить положение (для статической сцены)
    move(coords_t)
        Переместить к данному положению (для динамической сцены)
    reset(is_dynamic=False, do_orientate=True)
        Сбросить сцену
    enable_synchronization()
        Включить режим синхронизации
    disable_synchronization()
        Выключить режим синхронизации
    """

    def __init__(self):
        sim.simxFinish(-1) # close all opened connections
        self.client=sim.simxStart('127.0.0.1',19999,True,True,5000,5) # connect
        self.is_connected = self.client != -1
        self.synchronous = False
        self.cameras = []
        self.joints = []
        self.joint_ranges = np.zeros((JOINT_COUNT,2), dtype=np.float32)
        self.default_pos = np.asarray([0, 0, 2.97, 2.62, 1.57, 0])
        self.scene_controller = 0
        self.stereo_matcher = cv.StereoBM_create(numDisparities=48, blockSize=11)

        if self.is_connected:
            _, self.scene_controller = sim.simxGetObjectHandle(self.client, SCENE_CONTROLLER_NAME, sim.simx_opmode_blocking)
            for i in range(1,3):
                _, id = sim.simxGetObjectHandle(self.client, STEREO_CAMERA_NAME + str(i), sim.simx_opmode_blocking)
                self.cameras.append(id)
            for i in range(1,JOINT_COUNT+1):
                _, id = sim.simxGetObjectHandle(self.client, JOINT_NAME + str(i), sim.simx_opmode_blocking)
                self.joints.append(id)
                _, _, min_max, _, _ = sim.simxCallScriptFunction(self.client, SCENE_CONTROLLER_NAME, sim.sim_scripttype_childscript, JOINT_RANGE_FUNC_NAME,
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
        """Получить визуальную обратную связь

        Returns
        -------
        ndarray
            Изображение с камеры в RGB
        ndarray
            Карта глубины стереоизображения
        """

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
        """Получить обратную связь по положению

        Returns
        -------
        ndarray
            Обобщённые координаты
        """
        pos = np.zeros((JOINT_COUNT))
        for i in range(JOINT_COUNT):
            _, pos[i] = sim.simxGetJointPosition(self.client, self.joints[i], sim.simx_opmode_blocking)
        return self.clip_position(pos)

    def clip_position(self, coords):
        """Привести координаты к допустимому диапазону

        Parameters
        ----------
        coords : ndarray
            "Небезопасные" обобщённые координаты

        Returns
        -------
        ndarray
            "Безопасные" обобщённые координаты
        """

        return np.clip(coords, self.joint_ranges[:,0], self.joint_ranges[:,1])

    def set_target_position(self, coords):
        """Установить целевое положение (для динамической сцены)

        Parameters
        ----------
        coords : ndarray
            Обобщённые координаты
        """

        sim.simxPauseCommunication(self.client, 1)
        for i in range(JOINT_COUNT):
            sim.simxSetJointTargetPosition(self.client, self.joints[i], coords[i], sim.simx_opmode_oneshot)
        sim.simxPauseCommunication(self.client, 0)

    def set_position(self, coords):
        """Установить положение (для статической сцены)

        Не работает для динамической сцены. Для статической вызовает мгновенное перемещение.
        При работе с обратной связью следует использовать в режиме синхронизации для предотвращения
        получения неактуальных данных.

        Parameters
        ----------
        coords : ndarray
            Обобщённые координаты
        """

        sim.simxPauseCommunication(self.client, 1)
        for i in range(JOINT_COUNT):
            sim.simxSetJointPosition(self.client, self.joints[i], coords[i], sim.simx_opmode_oneshot)
        sim.simxPauseCommunication(self.client, 0)
        if self.synchronous:
            self._step_simulation()

    def move(self, coords_t):
        """Переместить к данному положению (для динамической сцены).

        Не поддерживается в режиме синхронизации. Задаёт целевое положение и ожидает его достижения
        через обратную связь. Функция завершит выполнение, если цель не может быть достигнута.

        Parameters
        ----------
        coords_t : ndarray
            Обобщённые координаты

        Returns
        -------
        bool
            Положение было достигнуто

        Raises
        ------
        ValueError
            Не поддерживается в режиме синхронизации
        """

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
        """Сбросить сцену

        Объекты сцены приобретают случайное положение в рабочей зоне манипулятора.
        Робот становится в начальную конфигурацию. Если do_orientate=True, то первая ось
        направляется в сторону объекта, иначе - принимает случайное значение в рабочем диапазоне.

        Parameters
        ----------
        is_dynamic : bool, optional
            Динамическая ли сцена, by default False
        do_orientate : bool, optional
            Следует ли повернуть робота в сторону объекта, by default True
        """

        # phi - угол сектора, в котором был сгенерирован целевой объект
        _, _, phi, _, _ = sim.simxCallScriptFunction(self.client, SCENE_CONTROLLER_NAME, sim.sim_scripttype_childscript, RESET_FUNC_NAME,
                                    [],[],[],bytearray(), sim.simx_opmode_blocking)
        pos = self.default_pos.copy()
        if do_orientate:
            offset = random.random()*0.2 - 0.1 # небольшое отклонение в зоне видимости объекта
            pos[0] = phi[0] + offset
        else:
            pos[0] = random.random()*(self.joint_ranges[0,1]-self.joint_ranges[0,0]) + self.joint_ranges[0,0]
        
        if is_dynamic:
            self.move(pos)
        else:
            self.set_position(pos)

    def enable_synchronization(self):
        """Включить режим синхронизации
        """
        sim.simxSynchronous(self.client,True)
        self.synchronous = True

    def disable_synchronization(self):
        """Выключить режим синхронизации
        """
        sim.simxSynchronous(self.client,False)
        self.synchronous =False

    def _step_simulation(self):
        """Шаг симуляции для режима синхронизации
        """
        sim.simxSynchronousTrigger(self.client)
        sim.simxGetPingTime(self.client)