# -*- coding: utf-8 -*-
import sys
import math
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QGroupBox, QGridLayout,
    QRadioButton, QSizePolicy
)
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtCore import Qt, QPointF, QTimer


# --- Математические утилиты для 3D ---

def create_identity_matrix():
    """Создает единичную матрицу 4x4."""
    return [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]


def create_translation_matrix(tx, ty, tz):
    """Создает матрицу трансляции."""
    return [
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ]


def create_scale_matrix(sx, sy, sz):
    """Создает матрицу масштабирования."""
    return [
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ]


def create_rotation_x_matrix(angle_rad):
    """Создает матрицу поворота вокруг оси X."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return [
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ]


def create_rotation_y_matrix(angle_rad):
    """Создает матрицу поворота вокруг оси Y."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return [
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ]


def create_rotation_z_matrix(angle_rad):
    """Создает матрицу поворота вокруг оси Z."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return [
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]


def create_reflection_matrix_xy():  # Отражение относительно плоскости XY (ось Z инвертируется)
    return create_scale_matrix(1, 1, -1)


def create_reflection_matrix_xz():  # Отражение относительно плоскости XZ (ось Y инвертируется)
    return create_scale_matrix(1, -1, 1)


def create_reflection_matrix_yz():  # Отражение относительно плоскости YZ (ось X инвертируется)
    return create_scale_matrix(-1, 1, 1)


def multiply_matrix_vector(matrix, vector):
    """Умножает матрицу 4x4 на вектор (x, y, z, w)."""
    result = [0, 0, 0, 0]
    for i in range(4):
        for j in range(4):
            result[i] += matrix[i][j] * vector[j]
    return result


def multiply_matrices(m1, m2):
    """Умножает две матрицы 4x4 (m1 * m2)."""
    result = create_identity_matrix()
    for i in range(4):
        for j in range(4):
            sum_val = 0
            for k in range(4):
                sum_val += m1[i][k] * m2[k][j]
            result[i][j] = sum_val
    return result


def perspective_projection_matrix(fov_y_rad, aspect_ratio, near, far):
    """Создает матрицу перспективной проекции."""
    # f = 1.0 / math.tan(fov_y_rad / 2.0) # Не используется в данной форме
    # Используем формулу с глубиной в диапазоне [-1, 1] для z' (OpenGL style)
    # или [0, 1] (DirectX style). Здесь будет [-1, 1] если z_ndc = clip_z / clip_w
    # Для простоты и чтобы избежать деления на ноль при fov=0 или aspect=0,
    # можно использовать более прямую формулу, если известны left, right, bottom, top на near plane.
    # Или, более стандартно:

    f = 1.0 / math.tan(fov_y_rad / 2.0)
    if far == near:  # Предотвращение деления на ноль
        z_range = 0.001
    else:
        z_range = far - near

    m = create_identity_matrix()
    m[0][0] = f / aspect_ratio
    m[1][1] = f
    m[2][2] = -(far + near) / z_range  # Переделано для отображения z в NDC
    m[2][3] = -(2 * far * near) / z_range
    m[3][2] = -1  # Для переноса w в z_clip
    m[3][3] = 0
    return m


def look_at_matrix(eye, target, up):
    """Создает матрицу вида (view matrix)."""
    # Вычисляем оси камеры
    # z_axis = normalize(eye - target) # В OpenGL камера смотрит в -Z
    z_axis_val = [eye[i] - target[i] for i in range(3)]

    # Если eye и target совпадают, z_axis будет нулевым, что приведет к ошибке.
    # Добавим небольшое смещение, если это так.
    if all(v == 0 for v in z_axis_val):
        z_axis_val[2] = 0.001  # Небольшое смещение, чтобы избежать деления на ноль

    z_axis_len = math.sqrt(sum(c * c for c in z_axis_val))
    z_axis = [c / z_axis_len for c in z_axis_val]

    # x_axis = normalize(cross_product(up, z_axis))
    x_axis_val = [
        up[1] * z_axis[2] - up[2] * z_axis[1],
        up[2] * z_axis[0] - up[0] * z_axis[2],
        up[0] * z_axis[1] - up[1] * z_axis[0]
    ]
    x_axis_len = math.sqrt(sum(c * c for c in x_axis_val))
    # Если up и z_axis коллинеарны
    if x_axis_len < 1e-6:  # Почти коллинеарны
        # Пытаемся использовать другой 'up' вектор, например, (0,0,1) если текущий 'up' был (0,1,0) и z_axis тоже вертикален
        # Или просто выбрать ортогональный к z_axis
        if abs(z_axis[1]) > 0.99:  # z_axis почти вертикальный
            temp_up = [1, 0, 0]
        else:
            temp_up = [0, 1, 0]
        x_axis_val = [
            temp_up[1] * z_axis[2] - temp_up[2] * z_axis[1],
            temp_up[2] * z_axis[0] - temp_up[0] * z_axis[2],
            temp_up[0] * z_axis[1] - temp_up[1] * z_axis[0]
        ]
        x_axis_len = math.sqrt(sum(c * c for c in x_axis_val))

    x_axis = [c / x_axis_len for c in x_axis_val]

    # y_axis = cross_product(z_axis, x_axis)
    y_axis_val = [
        z_axis[1] * x_axis[2] - z_axis[2] * x_axis[1],
        z_axis[2] * x_axis[0] - z_axis[0] * x_axis[2],
        z_axis[0] * x_axis[1] - z_axis[1] * x_axis[0]
    ]
    y_axis_len = math.sqrt(sum(c * c for c in y_axis_val))
    y_axis = [c / y_axis_len for c in y_axis_val]

    # Матрица вида View = R_inv * T_inv
    # R_inv (т.к. оси ортонормированы, R_inv = R_transpose)
    # T_inv переносит eye в начало координат

    view_matrix = [
        [x_axis[0], x_axis[1], x_axis[2], -sum(x_axis[i] * eye[i] for i in range(3))],
        [y_axis[0], y_axis[1], y_axis[2], -sum(y_axis[i] * eye[i] for i in range(3))],
        [z_axis[0], z_axis[1], z_axis[2], -sum(z_axis[i] * eye[i] for i in range(3))],
        [0, 0, 0, 1]
    ]
    return view_matrix


# --- Геометрия букв ---
def get_letter_B_geometry(W, H, D):
    """Возвращает вершины и ребра для буквы 'В'."""
    # Нормализованные координаты для передней грани (0-1 по X, 0-1 по Y)
    norm_front_vertices_B = [
        (0, 1), (0.3, 1), (1, 1), (1, 0.6), (0.3, 0.6),  # 0-4
        (0.3, 0.5),  # 5 (центральная перемычка на стойке, верхняя точка)
        (0.3, 0.4),  # 6 (центральная перемычка на стойке, нижняя точка)
        (1, 0.4), (1, 0), (0.3, 0), (0, 0),  # 7-10
        # Верхнее отверстие
        (0.4, 0.9), (0.9, 0.9), (0.9, 0.7), (0.4, 0.7),  # 11-14
        # Нижнее отверстие
        (0.4, 0.3), (0.9, 0.3), (0.9, 0.1), (0.4, 0.1)  # 15-18
    ]

    front_vertices = [(x * W, y * H, 0) for x, y in norm_front_vertices_B]
    back_vertices = [(x * W, y * H, D) for x, y in norm_front_vertices_B]

    all_vertices = front_vertices + back_vertices

    # Ребра для передней грани
    edges_front_B = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 1),  # Внешний контур верхнего полукольца и соединение со стойкой
        (4, 5),  # Часть стойки
        (5, 3),
        # Перемычка верхнего полукольца (внутренняя) - нет, (4,5) было частью стойки. (3,4) внешний контур. (1,4) замыкает верхнюю часть стойки.
        # (2,3) внешняя правая кромка верхнего полукольца
        # (3,?) внешняя правая кромка до середины

        # Внешний контур
        (0, 1), (1, 2), (2, 3), (3, 7), (7, 8), (8, 9), (9, 10), (10, 0),
        # Стойка и перемычки
        (1, 4), (4, 5), (5, 6), (6, 9), (4, 3), (6, 7),
        # (4,5) и (5,6) - разделение на стойке. (4,3) и (6,7) - перемычки к правой части
        # Верхнее отверстие
        (11, 12), (12, 13), (13, 14), (14, 11),
        # Нижнее отверстие
        (15, 16), (16, 17), (17, 18), (18, 15)
    ]

    all_edges = []
    all_edges.extend(edges_front_B)

    offset = len(norm_front_vertices_B)
    all_edges.extend([(i + offset, j + offset) for i, j in edges_front_B])  # Задние ребра

    for i in range(len(norm_front_vertices_B)):  # Ребра глубины
        all_edges.append((i, i + offset))

    return all_vertices, all_edges


def get_letter_D_geometry(W, H, D):
    """Возвращает вершины и ребра для буквы 'Д'."""
    norm_front_vertices_D = [
        (0.2, 0), (0.3, 0), (0.3, 0.4), (0.2, 0.4),  # Левая ножка (0-3)
        (0.7, 0), (0.8, 0), (0.8, 0.4), (0.7, 0.4),  # Правая ножка (4-7)
        (0, 0.4), (1, 0.4), (0.8, 1), (0.2, 1)  # "Столешница" (8-11)
    ]

    front_vertices = [(x * W, y * H, 0) for x, y in norm_front_vertices_D]
    back_vertices = [(x * W, y * H, D) for x, y in norm_front_vertices_D]

    all_vertices = front_vertices + back_vertices

    edges_front_D = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Левая ножка
        (4, 5), (5, 6), (6, 7), (7, 4),  # Правая ножка
        (8, 9), (9, 10), (10, 11), (11, 8)  # Столешница
    ]

    all_edges = []
    all_edges.extend(edges_front_D)

    offset = len(norm_front_vertices_D)
    all_edges.extend([(i + offset, j + offset) for i, j in edges_front_D])  # Задние ребра

    for i in range(len(norm_front_vertices_D)):  # Ребра глубины
        all_edges.append((i, i + offset))

    return all_vertices, all_edges


# --- Класс для 3D объекта ---
class Object3D:
    def __init__(self, name, base_vertices_func, initial_W, initial_H, initial_D):
        self.name = name
        self.base_vertices_func = base_vertices_func
        self.W, self.H, self.D = initial_W, initial_H, initial_D
        self.vertices = []  # Локальные координаты вершин (x,y,z)
        self.edges = []  # Пары индексов вершин
        self.model_matrix = create_identity_matrix()
        self.update_geometry()

    def update_geometry(self):
        raw_vertices, self.edges = self.base_vertices_func(self.W, self.H, self.D)
        # Переводим в формат (x,y,z,1) для матричных операций
        self.vertices = [[v[0], v[1], v[2], 1] for v in raw_vertices]

    def set_dimensions(self, W, H, D):
        self.W, self.H, self.D = W, H, D
        self.update_geometry()

    def transform(self, transform_matrix):
        self.model_matrix = multiply_matrices(transform_matrix, self.model_matrix)

    def reset_transform(self):
        self.model_matrix = create_identity_matrix()


# --- Виджет для отрисовки ---
class RenderWidget(QWidget):
    def __init__(self, scene, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.auto_scale_factor = 1.0
        self.auto_translate_x = 0.0
        self.auto_translate_y = 0.0
        self.needs_auto_scale_recalc = True

    def project_vertex(self, vertex_local, model_matrix, view_matrix, projection_matrix):
        # Локальные -> Мировые -> Видовые -> Клиппинг -> NDC -> Экранные

        # Локальные в мировые (с учетом модели)
        vertex_world = multiply_matrix_vector(model_matrix, vertex_local)

        # Мировые в видовые (с учетом камеры)
        vertex_view = multiply_matrix_vector(view_matrix, vertex_world)

        # Видовые в клиппинг (проекция)
        vertex_clip = multiply_matrix_vector(projection_matrix, vertex_view)

        # Перспективное деление (в NDC - Normalized Device Coordinates)
        w_clip = vertex_clip[3]
        if abs(w_clip) < 1e-6:  # Точка слишком далеко или за камерой (вдоль оси w)
            return None  # Не рисуем эту точку

        # Проверка, находится ли точка в пределах отсечения по Z (после перспективного деления)
        # z_ndc = vertex_clip[2] / w_clip
        # if not (-1 <= z_ndc <= 1): # Если используется такая система NDC
        #     return None # Отсечение по глубине (простое)
        # Проверка на отсечение по frustum более сложна для ребер, здесь упрощено.
        # Для проволочной модели часто рисуют все, что не "слишком далеко".

        # Если точка за ближней плоскостью отсечения (near plane), это может вызвать проблемы
        # w_clip < 0 означает, что точка за камерой.
        # near plane в проекции обычно соответствует z_ndc = -1 или z_ndc = 0
        # vertex_view[2] - z координата в пространстве камеры. Если > -near_plane_dist (для камеры смотрящей в -Z)
        # то точка перед камерой.
        # TODO: Добавить более корректное отсечение ребер по frustum (например, Liang-Barsky или Sutherland-Hodgman для 3D)
        # Для простоты, пока не отсекаем жестко, но проверяем w_clip.

        x_ndc = vertex_clip[0] / w_clip
        y_ndc = vertex_clip[1] / w_clip
        # z_ndc = vertex_clip[2] / w_clip # z_ndc используется для Z-буфера, здесь не нужен для 2D координат

        # Преобразование из NDC [-1, 1] в экранные координаты [0, width] и [0, height]
        # Y инвертируется, так как в экранных координатах Y растет вниз
        screen_x = (x_ndc + 1) * 0.5 * self.width()
        screen_y = (1 - y_ndc) * 0.5 * self.height()

        return QPointF(screen_x, screen_y)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("lightgray"))  # Фон
        painter.setPen(QPen(QColor("black"), 1))

        view_matrix = self.scene.get_view_matrix()
        projection_matrix = self.scene.get_projection_matrix(self.width() / max(1, self.height()))

        all_projected_points_for_autoscale = []

        # Фаза 1: Проекция всех видимых вершин для авто масштабирования (если нужно)
        # (Пропущено для простоты в этом примере, авто-масштабирование может быть сложным)
        # Сейчас просто рисуем как есть. "Автомасштабирование" в требовании может означать
        # что объекты должны быть видны при начальных параметрах, или viewport transform.

        for obj in self.scene.objects:
            for edge in obj.edges:
                v1_local = obj.vertices[edge[0]]
                v2_local = obj.vertices[edge[1]]

                p1_screen = self.project_vertex(v1_local, obj.model_matrix, view_matrix, projection_matrix)
                p2_screen = self.project_vertex(v2_local, obj.model_matrix, view_matrix, projection_matrix)

                if p1_screen and p2_screen:
                    painter.drawLine(p1_screen, p2_screen)

    def resizeEvent(self, event):
        self.needs_auto_scale_recalc = True
        super().resizeEvent(event)


# --- Класс для сцены ---
class Scene:
    def __init__(self):
        self.objects = []
        # Параметры камеры
        self.camera_eye = [150, 150, 300]  # Позиция камеры
        self.camera_target = [0, 75, 0]  # Точка, на которую смотрит камера
        self.camera_up = [0, 1, 0]  # Направление "вверх" для камеры
        self.fov_y_deg = 45  # Угол обзора по Y в градусах
        self.near_plane = 10
        self.far_plane = 1000

    def add_object(self, obj):
        self.objects.append(obj)

    def get_view_matrix(self):
        return look_at_matrix(self.camera_eye, self.camera_target, self.camera_up)

    def get_projection_matrix(self, aspect_ratio):
        fov_y_rad = math.radians(self.fov_y_deg)
        return perspective_projection_matrix(fov_y_rad, aspect_ratio, self.near_plane, self.far_plane)

    # Методы для управления камерой
    def move_camera(self, dx, dy, dz):
        self.camera_eye[0] += dx
        self.camera_eye[1] += dy
        self.camera_eye[2] += dz
        # self.camera_target[0] += dx # Если таргет тоже двигается с камерой
        # self.camera_target[1] += dy
        # self.camera_target[2] += dz

    def rotate_camera(self, pitch_rad, yaw_rad):
        # Вращение камеры вокруг target (орбитальное) или вокруг своей оси (FPS-style)
        # Здесь реализуем орбитальное вращение вокруг target

        # Вектор от target к eye
        direction = [self.camera_eye[i] - self.camera_target[i] for i in range(3)]

        # Горизонтальное вращение (yaw) вокруг оси UP мировой системы (self.camera_up)
        # Это упрощенное вращение вокруг глобальной Y оси
        x, y, z = direction[0], direction[1], direction[2]

        # Yaw (вокруг Y)
        new_x_yaw = x * math.cos(yaw_rad) + z * math.sin(yaw_rad)
        new_z_yaw = -x * math.sin(yaw_rad) + z * math.cos(yaw_rad)
        direction[0], direction[2] = new_x_yaw, new_z_yaw

        # Pitch (вокруг оси X камеры, которая перпендикулярна forward и up)
        # Получаем ось X камеры (право)
        forward = [-d for d in direction]  # Направление взгляда от eye к target
        forward_len = math.sqrt(sum(c * c for c in forward))
        if forward_len > 1e-6:
            forward = [c / forward_len for c in forward]

        right_val = [
            self.camera_up[1] * forward[2] - self.camera_up[2] * forward[1],
            self.camera_up[2] * forward[0] - self.camera_up[0] * forward[2],
            self.camera_up[0] * forward[1] - self.camera_up[1] * forward[0]
        ]
        right_len = math.sqrt(sum(c * c for c in right_val))
        if right_len > 1e-6:
            right = [c / right_len for c in right_val]
        else:  # Если forward и up коллинеарны (смотрим строго вверх/вниз)
            right = [1, 0, 0]  # Используем мировую X

        # Матрица поворота вокруг оси right
        c, s = math.cos(pitch_rad), math.sin(pitch_rad)
        R_pitch = [  # Rodrigues' rotation formula or quaternion
            [c + right[0] ** 2 * (1 - c), right[0] * right[1] * (1 - c) - right[2] * s,
             right[0] * right[2] * (1 - c) + right[1] * s],
            [right[1] * right[0] * (1 - c) + right[2] * s, c + right[1] ** 2 * (1 - c),
             right[1] * right[2] * (1 - c) - right[0] * s],
            [right[2] * right[0] * (1 - c) - right[1] * s, right[2] * right[1] * (1 - c) + right[0] * s,
             c + right[2] ** 2 * (1 - c)]
        ]

        x, y, z = direction[0], direction[1], direction[2]
        direction[0] = R_pitch[0][0] * x + R_pitch[0][1] * y + R_pitch[0][2] * z
        direction[1] = R_pitch[1][0] * x + R_pitch[1][1] * y + R_pitch[1][2] * z
        direction[2] = R_pitch[2][0] * x + R_pitch[2][1] * y + R_pitch[2][2] * z

        self.camera_eye = [self.camera_target[i] + direction[i] for i in range(3)]


# --- Главное окно ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторная работа №2 - Каркасная модель")
        self.setGeometry(100, 100, 1000, 700)

        self.scene = Scene()
        self.object_B = Object3D("B", get_letter_B_geometry, 80, 100, 20)
        self.object_D = Object3D("D", get_letter_D_geometry, 80, 100, 20)

        # Размещение буквы Д рядом с B
        initial_translation_D = create_translation_matrix(self.object_B.W + 20, 0, 0)  # 20 - отступ
        self.object_D.transform(initial_translation_D)

        self.scene.add_object(self.object_B)
        self.scene.add_object(self.object_D)
        self.selected_object = self.object_B  # По умолчанию выбрана первая буква

        self.render_widget = RenderWidget(self.scene)

        # Элементы управления
        controls_layout = QVBoxLayout()

        # Выбор объекта
        obj_select_group = QGroupBox("Выбор объекта")
        obj_select_layout = QHBoxLayout()
        self.rb_B = QRadioButton("Буква В")
        self.rb_B.setChecked(True)
        self.rb_B.toggled.connect(lambda: self.select_object(self.object_B))
        self.rb_D = QRadioButton("Буква Д")
        self.rb_D.toggled.connect(lambda: self.select_object(self.object_D))
        obj_select_layout.addWidget(self.rb_B)
        obj_select_layout.addWidget(self.rb_D)
        obj_select_group.setLayout(obj_select_layout)
        controls_layout.addWidget(obj_select_group)

        # Параметры букв
        params_group = QGroupBox("Параметры выбранной буквы")
        params_layout = QGridLayout()
        params_layout.addWidget(QLabel("Ширина (W):"), 0, 0)
        self.width_input = QDoubleSpinBox()
        self.width_input.setRange(10, 500);
        self.width_input.setValue(self.selected_object.W)
        params_layout.addWidget(self.width_input, 0, 1)

        params_layout.addWidget(QLabel("Высота (H):"), 1, 0)
        self.height_input = QDoubleSpinBox()
        self.height_input.setRange(10, 500);
        self.height_input.setValue(self.selected_object.H)
        params_layout.addWidget(self.height_input, 1, 1)

        params_layout.addWidget(QLabel("Глубина (D):"), 2, 0)
        self.depth_input = QDoubleSpinBox()
        self.depth_input.setRange(10, 500);
        self.depth_input.setValue(self.selected_object.D)
        params_layout.addWidget(self.depth_input, 2, 1)

        apply_dims_button = QPushButton("Применить размеры")
        apply_dims_button.clicked.connect(self.apply_dimensions)
        params_layout.addWidget(apply_dims_button, 3, 0, 1, 2)
        params_group.setLayout(params_layout)
        controls_layout.addWidget(params_group)

        # Управление объектом
        obj_transform_group = QGroupBox("Трансформации объекта")
        obj_transform_layout = QGridLayout()
        self.step_translate_obj_input = QDoubleSpinBox();
        self.step_translate_obj_input.setValue(10);
        self.step_translate_obj_input.setSingleStep(1)
        self.step_rotate_obj_input = QDoubleSpinBox();
        self.step_rotate_obj_input.setValue(15);
        self.step_rotate_obj_input.setSingleStep(1)  # в градусах

        obj_transform_layout.addWidget(QLabel("Шаг смещ.:"), 0, 0)
        obj_transform_layout.addWidget(self.step_translate_obj_input, 0, 1)
        obj_transform_layout.addWidget(QLabel("Шаг повор.(°):"), 0, 2)
        obj_transform_layout.addWidget(self.step_rotate_obj_input, 0, 3)

        buttons_obj = [
            ("Смест. X+", lambda: self.translate_selected_object(self.step_translate_obj_input.value(), 0, 0)),
            ("Смест. X-", lambda: self.translate_selected_object(-self.step_translate_obj_input.value(), 0, 0)),
            ("Смест. Y+", lambda: self.translate_selected_object(0, self.step_translate_obj_input.value(), 0)),
            ("Смест. Y-", lambda: self.translate_selected_object(0, -self.step_translate_obj_input.value(), 0)),
            ("Смест. Z+", lambda: self.translate_selected_object(0, 0, self.step_translate_obj_input.value())),
            ("Смест. Z-", lambda: self.translate_selected_object(0, 0, -self.step_translate_obj_input.value())),
            ("Повер. X+", lambda: self.rotate_selected_object(math.radians(self.step_rotate_obj_input.value()), 0, 0)),
            ("Повер. X-", lambda: self.rotate_selected_object(math.radians(-self.step_rotate_obj_input.value()), 0, 0)),
            ("Повер. Y+", lambda: self.rotate_selected_object(0, math.radians(self.step_rotate_obj_input.value()), 0)),
            ("Повер. Y-", lambda: self.rotate_selected_object(0, math.radians(-self.step_rotate_obj_input.value()), 0)),
            ("Повер. Z+", lambda: self.rotate_selected_object(0, 0, math.radians(self.step_rotate_obj_input.value()))),
            ("Повер. Z-", lambda: self.rotate_selected_object(0, 0, math.radians(-self.step_rotate_obj_input.value()))),
            ("Сброс", self.reset_selected_object_transform),
            ("Отр. XY", lambda: self.reflect_selected_object("xy")),
            ("Отр. XZ", lambda: self.reflect_selected_object("xz")),
            ("Отр. YZ", lambda: self.reflect_selected_object("yz")),
        ]
        row, col = 1, 0
        for name, action in buttons_obj:
            btn = QPushButton(name);
            btn.clicked.connect(action)
            obj_transform_layout.addWidget(btn, row, col)
            col += 1
            if col > 3: col = 0; row += 1
        obj_transform_group.setLayout(obj_transform_layout)
        controls_layout.addWidget(obj_transform_group)

        # Управление камерой
        cam_transform_group = QGroupBox("Трансформации камеры")
        cam_transform_layout = QGridLayout()
        self.step_translate_cam_input = QDoubleSpinBox();
        self.step_translate_cam_input.setValue(20);
        self.step_translate_cam_input.setSingleStep(1)
        self.step_rotate_cam_input = QDoubleSpinBox();
        self.step_rotate_cam_input.setValue(5);
        self.step_rotate_cam_input.setSingleStep(1)  # в градусах

        cam_transform_layout.addWidget(QLabel("Шаг смещ.:"), 0, 0)
        cam_transform_layout.addWidget(self.step_translate_cam_input, 0, 1)
        cam_transform_layout.addWidget(QLabel("Шаг повор.(°):"), 0, 2)
        cam_transform_layout.addWidget(self.step_rotate_cam_input, 0, 3)

        buttons_cam = [
            ("Кам. X+", lambda: self.scene.move_camera(self.step_translate_cam_input.value(), 0, 0)),
            ("Кам. X-", lambda: self.scene.move_camera(-self.step_translate_cam_input.value(), 0, 0)),
            ("Кам. Y+", lambda: self.scene.move_camera(0, self.step_translate_cam_input.value(), 0)),
            ("Кам. Y-", lambda: self.scene.move_camera(0, -self.step_translate_cam_input.value(), 0)),
            ("Кам. Z+", lambda: self.scene.move_camera(0, 0, self.step_translate_cam_input.value())),
            # Zoom In (приближение)
            ("Кам. Z-", lambda: self.scene.move_camera(0, 0, -self.step_translate_cam_input.value())),
            # Zoom Out (отдаление)
            ("Кам. Pitch+", lambda: self.scene.rotate_camera(math.radians(self.step_rotate_cam_input.value()), 0)),
            ("Кам. Pitch-", lambda: self.scene.rotate_camera(math.radians(-self.step_rotate_cam_input.value()), 0)),
            ("Кам. Yaw+", lambda: self.scene.rotate_camera(0, math.radians(self.step_rotate_cam_input.value()))),
            ("Кам. Yaw-", lambda: self.scene.rotate_camera(0, math.radians(-self.step_rotate_cam_input.value()))),
        ]
        row, col = 1, 0
        for name, action in buttons_cam:
            btn = QPushButton(name)
            btn.clicked.connect(action)
            btn.clicked.connect(self.render_widget.update)  # Обновление после каждого действия
            cam_transform_layout.addWidget(btn, row, col)
            col += 1
            if col > 1: col = 0; row += 1  # По 2 кнопки в ряд для камеры
        cam_transform_group.setLayout(cam_transform_layout)
        controls_layout.addWidget(cam_transform_group)

        controls_layout.addStretch()  # Растягивающийся элемент для выравнивания

        # Основной макет окна
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.render_widget, 3)  # Область рендера занимает больше места

        controls_container = QWidget()
        controls_container.setLayout(controls_layout)
        main_layout.addWidget(controls_container, 1)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Таймер для обновления (если нужно для анимации, здесь не используется)
        # self.timer = QTimer(self)
        # self.timer.timeout.connect(self.render_widget.update)
        # self.timer.start(16) # ~60 FPS

    def select_object(self, obj):
        self.selected_object = obj
        self.width_input.setValue(obj.W)
        self.height_input.setValue(obj.H)
        self.depth_input.setValue(obj.D)
        self.render_widget.update()

    def apply_dimensions(self):
        if self.selected_object:
            W = self.width_input.value()
            H = self.height_input.value()
            D = self.depth_input.value()
            self.selected_object.set_dimensions(W, H, D)
            self.render_widget.update()

    def translate_selected_object(self, dx, dy, dz):
        if self.selected_object:
            # Матрица трансляции должна применяться к текущей матрице модели
            # T_new * M_old
            # Однако, если мы хотим транслировать в мировых координатах, то M_old * T_new
            # Для трансляции в локальных координатах объекта: T_new * M_old
            # Для трансляции в мировых: M_new = T_world * M_old
            # Здесь проще всего добавлять трансляцию к существующей матрице модели M_new = M_translate * M_current
            # Что соответствует трансляции в ЛОКАЛЬНЫХ координатах после всех предыдущих поворотов/масштабирований
            # Если нужна трансляция в МИРОВЫХ, то M_new = M_current * M_translate_world
            # Проще: M_new = T * M_old (пре-умножение) - смещение в глобальных осях до поворота объекта
            # M_new = M_old * T (пост-умножение) - смещение в локальных осях после поворота объекта
            # Обычно интуитивно ожидают пост-умножение для локальных осей.
            # Но для кнопок "Смест X+" часто ожидают смещение вдоль мировой оси X.
            # Давайте сделаем смещение в мировых координатах (пост-умножение на матрицу модели):
            # model_matrix = model_matrix * translation_matrix
            # Это неверно. Translation (dx,dy,dz) всегда в той системе координат, где он определен.
            # M' = T(dx,dy,dz) * M. Вершины V' = M' * V_local = T * M * V_local. Это транслирует уже преобразованный объект.

            # Правильно для добавления трансляции:
            # M_obj_world = M_translate_world * M_obj_world
            # Или, если хотим двигать объект вдоль его локальных осей после поворота:
            # M_obj_world = M_obj_world * M_translate_local

            # Для простоты, будем считать, что tx, ty, tz это смещения в мировых осях.
            # Тогда мы должны добавить их к компонентам трансляции в model_matrix.
            # model_matrix[0][3] += dx; model_matrix[1][3] += dy; model_matrix[2][3] += dz;
            # Это проще, чем умножать матрицы, если мы уверены, что нет масштабирования/поворота, влияющего на трансляцию.
            # Безопаснее через умножение:

            # Вариант 1: Трансляция в мировых осях (объект сдвигается вдоль осей мира)
            # M_new = T_world * M_current
            # t_matrix = create_translation_matrix(dx, dy, dz)
            # self.selected_object.model_matrix = multiply_matrices(t_matrix, self.selected_object.model_matrix)

            # Вариант 2: Трансляция вдоль локальных осей объекта (объект сдвигается "вперед", "вправо" относительно себя)
            # M_new = M_current * T_local
            t_matrix = create_translation_matrix(dx, dy, dz)
            self.selected_object.model_matrix = multiply_matrices(self.selected_object.model_matrix, t_matrix)

            self.render_widget.update()

    def rotate_selected_object(self, angle_x_rad, angle_y_rad, angle_z_rad):
        if self.selected_object:
            # Вращение вокруг локальных осей объекта
            # M_new = M_current * R_x * R_y * R_z (или другой порядок)
            # Вращение вокруг мировых осей (с центром в начале координат мира)
            # M_new = R_z * R_y * R_x * M_current
            # Для вращения объекта вокруг его собственного центра и локальных осей:
            # 1. Перенести в начало координат (если центр не там) - не нужно, если вращаем вокруг (0,0,0) локальной системы
            # 2. Повернуть
            # 3. Вернуть обратно
            # Проще: M_new = M_current * R_local

            # Поворот вокруг локальной оси X
            if angle_x_rad != 0:
                rx_matrix = create_rotation_x_matrix(angle_x_rad)
                self.selected_object.model_matrix = multiply_matrices(self.selected_object.model_matrix, rx_matrix)
            # Поворот вокруг локальной оси Y
            if angle_y_rad != 0:
                ry_matrix = create_rotation_y_matrix(angle_y_rad)
                self.selected_object.model_matrix = multiply_matrices(self.selected_object.model_matrix, ry_matrix)
            # Поворот вокруг локальной оси Z
            if angle_z_rad != 0:
                rz_matrix = create_rotation_z_matrix(angle_z_rad)
                self.selected_object.model_matrix = multiply_matrices(self.selected_object.model_matrix, rz_matrix)

            self.render_widget.update()

    def reflect_selected_object(self, plane_str):
        if self.selected_object:
            reflection_m = create_identity_matrix()
            if plane_str == "xy":  # Отражение по Z
                reflection_m = create_reflection_matrix_xy()
            elif plane_str == "xz":  # Отражение по Y
                reflection_m = create_reflection_matrix_xz()
            elif plane_str == "yz":  # Отражение по X
                reflection_m = create_reflection_matrix_yz()

            # Отражение применяется относительно локальных осей объекта
            # M_new = M_current * M_reflect
            self.selected_object.model_matrix = multiply_matrices(self.selected_object.model_matrix, reflection_m)
            self.render_widget.update()

    def reset_selected_object_transform(self):
        if self.selected_object:
            self.selected_object.reset_transform()
            # Если объект не B, нужно восстановить его начальное смещение
            if self.selected_object.name == "D":
                initial_translation_D = create_translation_matrix(self.object_B.W + 20, 0, 0)
                self.selected_object.transform(initial_translation_D)
            self.render_widget.update()

    def keyPressEvent(self, event):
        # Управление выбранным объектом с клавиатуры (по необходимости)
        # Пример:
        step_t = self.step_translate_obj_input.value()
        step_r_deg = self.step_rotate_obj_input.value()
        step_r_rad = math.radians(step_r_deg)

        key_handled = True
        if event.key() == Qt.Key_Left:
            self.translate_selected_object(-step_t, 0, 0)
        elif event.key() == Qt.Key_Right:
            self.translate_selected_object(step_t, 0, 0)
        elif event.key() == Qt.Key_Up:
            if event.modifiers() & Qt.ShiftModifier:
                self.translate_selected_object(0, 0, step_t)  # Вперед по Z
            else:
                self.translate_selected_object(0, step_t, 0)  # Вверх по Y
        elif event.key() == Qt.Key_Down:
            if event.modifiers() & Qt.ShiftModifier:
                self.translate_selected_object(0, 0, -step_t)  # Назад по Z
            else:
                self.translate_selected_object(0, -step_t, 0)  # Вниз по Y
        # Добавить вращение и управление камерой по желанию
        # Например, WASDQE для вращения объекта
        elif event.key() == Qt.Key_A:
            self.rotate_selected_object(0, -step_r_rad, 0)  # Yaw влево
        elif event.key() == Qt.Key_D:
            self.rotate_selected_object(0, step_r_rad, 0)  # Yaw вправо
        elif event.key() == Qt.Key_W:
            self.rotate_selected_object(-step_r_rad, 0, 0)  # Pitch вверх
        elif event.key() == Qt.Key_S:
            self.rotate_selected_object(step_r_rad, 0, 0)  # Pitch вниз
        elif event.key() == Qt.Key_Q:
            self.rotate_selected_object(0, 0, -step_r_rad)  # Roll влево
        elif event.key() == Qt.Key_E:
            self.rotate_selected_object(0, 0, step_r_rad)  # Roll вправо
        else:
            key_handled = False
            super().keyPressEvent(event)

        if key_handled:
            self.render_widget.update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec())