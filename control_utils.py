import numpy as np
import network_utils

SEARCH_RESOLUTION = 20 # разрешение поиска

class Searcher():
    """Класс, отвечающий за поиск объекта вдоль первой координаты.

    Attributes
    -------
    bounds : list or ndarray
        Минимальная и максимальная координаты
    res : int
        Разрешение поиска - количество позиций в диапазоне для проверки
    map : ndarray
        Результаты проверки позиций (-1 - не найден, 0 - неизвестно, 1 - найден)

    Methods
    -------
    get_coord(current_coord)
        Получить следующую координату
    mark_coord(coord, is_found)
        Отметить результат проверки по координате
    reset()
        Сбросить результаты проверки
    """

    def __init__(self, bounds):
        self.bounds = bounds
        self.res = SEARCH_RESOLUTION
        self.map = np.zeros((self.res))
    
    def get_coord(self, current_coord):
        """Получить следующую координату

        Выбрать координату с найденным объектом или ближайшую из непроверенных

        Parameters
        ----------
        current_coord : float
            Текущая координата

        Returns
        -------
        float
            Следующая координата
        """

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
        """Отметить результат проверки по координате

        Parameters
        ----------
        coord : float
            Координата
        is_found : bool
            Найден ли объект
        """

        index = self._get_index(coord)
        self.map[index] = 1 if is_found else -1

    def reset(self):
        """Сбросить результаты проверки
        """

        self.map = np.zeros((self.res))

    def _get_index(self, coord):
        index = self.res * (coord - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        return np.clip(int(index), 0, self.res-1)

    def _get_coord(self, index):
        return (self.bounds[1] - self.bounds[0]) * index / self.res + self.bounds[0]

def find_and_approach(robot, detector, control_network):
    """Осуществляет поиск цели вдоль рабочей зоны и приближение к ней.
    Не работает в статической сцене.

    Parameters
    ----------
    robot : Robot
        Управляемый робот, подключенный к сцене
    detector : ObjectDetector
        Детектор объекта
    control_network : tf.keras.Model
        Обученная управляющая нейронная сеть state->action

    Returns
    -------
    bool
        Удалось ли достигнуть целевой объект
    """

    # Приведение к стандартным значениям всех координат робота, кроме первой
    next_pos = robot.default_pos.copy()
    next_pos[0] = robot.get_adometry_feedback()[0]
    robot.move(next_pos)

    searcher = Searcher(robot.joint_ranges[0,:])
    step = 0
    is_searching = True
    while True:
        state = network_utils.get_state(robot, detector)
        area = state[3]*state[4] # площадь обрамляющего цель прямоугольника
        if area > 0.2:
            return True
        elif step >= 35:
            return False
        else:
            step+=1
            is_target_found = area > 0.001
            pos = network_utils.extract_pos(state)
            if is_searching:
                # Обновление карты присутствия цели только при управлении от searcher,
                # что гарантирует стандартные значения всех координат робота, кроме первой
                searcher.mark_coord(pos[0], is_target_found)
            
            # Вычисление следующей координаты
            if is_target_found:
                action = control_network(np.expand_dims(state, axis=0)).numpy()
                action = np.squeeze(action, axis=0)
                next_pos = robot.clip_position(pos + action)
                is_searching = False
            else:
                next_pos = robot.default_pos.copy()
                next_pos[0] = searcher.get_coord(pos[0])
                is_searching = True

            is_pos_reached = robot.move(next_pos)
            if (not is_pos_reached) and is_searching:
                # Сектор недосягаем, делаем предположение, что там нет цели
                searcher.mark_coord(next_pos[0], False)
    return False