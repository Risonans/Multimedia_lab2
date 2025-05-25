import sys
import math
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QSlider, QPushButton, QDoubleSpinBox, QComboBox,
    QSizePolicy
)
from PySide6.QtGui import QPainter, QColor, QPen, QPolygonF, QImage, QPixmap, QFont
from PySide6.QtCore import Qt, QPointF, Signal, Slot, QTimer


# --- Математические утилиты (трансформации) ---
# (Остаются без изменений, как в предыдущем ответе)
def translation_matrix(dx, dy, dz):
    return np.array([
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1]
    ], dtype=float)


def scaling_matrix(sx, sy, sz):
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ], dtype=float)


def rotation_matrix_x(angle_rad):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ], dtype=float)


def rotation_matrix_y(angle_rad):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ], dtype=float)


def rotation_matrix_z(angle_rad):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=float)


def perspective_projection_matrix(fov_deg, aspect_ratio, near, far):
    fov_rad = math.radians(fov_deg)
    f = 1.0 / math.tan(fov_rad / 2.0)
    return np.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ], dtype=float)


def look_at_matrix(eye, target, up):
    f = (target - eye)
    f_norm = np.linalg.norm(f)
    if f_norm < 1e-6:
        f = np.array([0, 0, -1])  # Если глаз и цель совпадают, смотрим по -Z
    else:
        f = f / f_norm

    s = np.cross(f, up)
    s_norm = np.linalg.norm(s)
    if s_norm < 1e-6:  # f и up коллинеарны. Выбрать s ортогонально f
        if abs(f[1]) > 0.99:  # f почти вертикален
            s = np.array([1, 0, 0])
        else:
            s = np.cross(f, np.array([0, 1, 0]))  # Попробовать с мировым Y
            s_norm = np.linalg.norm(s)
            if s_norm < 1e-6:  # Все еще коллинеарны (f параллелен мировому Y)
                s = np.cross(f, np.array([0, 0, 1]))  # Попробовать с мировым Z
                s_norm = np.linalg.norm(s)
            s = s / (s_norm if s_norm > 1e-6 else 1.0)

    else:
        s = s / s_norm

    u = np.cross(s, f)  # Уже должен быть нормирован

    mat = np.identity(4)
    mat[0, 0], mat[0, 1], mat[0, 2] = s[0], s[1], s[2]
    mat[1, 0], mat[1, 1], mat[1, 2] = u[0], u[1], u[2]
    mat[2, 0], mat[2, 1], mat[2, 2] = -f[0], -f[1], -f[2]
    mat[0, 3] = -np.dot(s, eye)
    mat[1, 3] = -np.dot(u, eye)
    mat[2, 3] = np.dot(f, eye)
    return mat


# --- Класс для 3D Объекта ---
class Letter3D:
    def __init__(self, base_vertices, base_faces, char_name=""):
        self.base_vertices = np.array(base_vertices, dtype=float)
        self.faces = np.array(base_faces, dtype=int)
        self.edges = self._generate_edges_from_faces(self.faces)

        self.height = 1.0
        self.width = 1.0  # Будет переопределено пропорциями буквы
        self.depth = 0.2  # Базовая толщина

        self.char_name = char_name

        self.translation = np.array([0.0, 0.0, 0.0])
        self.rotation_angles = np.array([0.0, 0.0, 0.0])  # X, Y, Z в градусах
        # self.obj_scale = np.array([1.0, 1.0, 1.0]) # Удалим, т.к. set_dimensions покрывает это

        self.current_vertices = self.base_vertices.copy()
        self.model_matrix = np.identity(4)

        # Определяем фактическую ширину базовой модели
        if self.base_vertices.shape[0] > 0:
            min_x_base = np.min(self.base_vertices[:, 0])
            max_x_base = np.max(self.base_vertices[:, 0])
            self.base_model_width = max_x_base - min_x_base
            min_y_base = np.min(self.base_vertices[:, 1])
            max_y_base = np.max(self.base_vertices[:, 1])
            self.base_model_height = max_y_base - min_y_base
            min_z_base = np.min(self.base_vertices[:, 2])
            max_z_base = np.max(self.base_vertices[:, 2])
            self.base_model_depth = max_z_base - min_z_base
        else:
            self.base_model_width = 1.0
            self.base_model_height = 1.0
            self.base_model_depth = 0.2  # Значение по умолчанию, если нет вершин

        self.width = self.base_model_width  # Начальная ширина соответствует базовой модели
        self.update_model_matrix()

    def _generate_edges_from_faces(self, faces):
        edges = set()
        for face in faces:
            for i in range(len(face)):
                v1_idx = face[i]
                v2_idx = face[(i + 1) % len(face)]
                edge = tuple(sorted((v1_idx, v2_idx)))
                edges.add(edge)
        return list(edges)

    def set_dimensions(self, height, width, depth):
        self.height = height
        self.width = width
        self.depth = depth
        self.update_model_matrix()

    def set_translation(self, tx, ty, tz):
        self.translation = np.array([tx, ty, tz])
        self.update_model_matrix()

    def set_rotation(self, rx_deg, ry_deg, rz_deg):
        self.rotation_angles = np.array([rx_deg, ry_deg, rz_deg])
        self.update_model_matrix()

    def update_model_matrix(self):
        # Масштабирование для приведения базовой модели к заданным self.width, self.height, self.depth
        # Учитываем, что base_vertices уже имеют свои размеры.
        # Например, если base_model_height = 1.0 (из-за нормализации при создании),
        # а self.height = 2.0, то scale_y = 2.0 / 1.0 = 2.0.
        scale_x = self.width / self.base_model_width if self.base_model_width > 1e-6 else 1.0
        scale_y = self.height / self.base_model_height if self.base_model_height > 1e-6 else 1.0
        scale_z = self.depth / self.base_model_depth if self.base_model_depth > 1e-6 else 1.0

        dim_scale_matrix = scaling_matrix(scale_x, scale_y, scale_z)

        trans_m = translation_matrix(*self.translation)
        rot_x_m = rotation_matrix_x(math.radians(self.rotation_angles[0]))
        rot_y_m = rotation_matrix_y(math.radians(self.rotation_angles[1]))
        rot_z_m = rotation_matrix_z(math.radians(self.rotation_angles[2]))

        self.model_matrix = trans_m @ rot_z_m @ rot_y_m @ rot_x_m @ dim_scale_matrix
        self.current_vertices = (self.model_matrix @ self.base_vertices.T).T


# --- Вспомогательная функция для создания 3D геометрии букв ---
def _create_extruded_geometry(
        all_front_xy_coords,  # Список всех 2D точек [(x,y),...] передней грани (включая точки дырок)
        front_face_triangles_idxs,  # Список троек индексов [(i,j,k),...] для триангуляции передней грани
        outer_contour_vertex_count,  # Количество вершин во внешнем контуре (они идут первыми в all_front_xy_coords)
        holes_vertex_counts,  # Список количеств вершин для каждой дыры
        base_depth_param  # Базовая глубина/толщина буквы
):
    """
    Создает 3D вершины и грани для экструдированной формы.
    Нормали граней будут направлены наружу.
    Центрирует модель по X и Y относительно ее габаритов и нормализует высоту до ~1.0.
    """
    if not all_front_xy_coords:
        return [], []

    # Нормализация и центрирование 2D координат
    coords_np = np.array(all_front_xy_coords, dtype=float)
    min_x, max_x = np.min(coords_np[:, 0]), np.max(coords_np[:, 0])
    min_y, max_y = np.min(coords_np[:, 1]), np.max(coords_np[:, 1])

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    current_height = max_y - min_y
    scale_factor = 1.0 / current_height if current_height > 1e-6 else 1.0

    normalized_xy_coords = []
    for x, y in all_front_xy_coords:
        norm_x = (x - center_x) * scale_factor
        norm_y = (y - center_y) * scale_factor
        normalized_xy_coords.append((norm_x, norm_y))

    base_vertices_3d = []
    # Создаем вершины передней грани (z = base_depth_param / 2)
    for x, y in normalized_xy_coords:
        base_vertices_3d.append([x, y, base_depth_param / 2.0, 1.0])

    num_total_front_vertices = len(normalized_xy_coords)

    # Создаем вершины задней грани (z = -base_depth_param / 2)
    for x, y in normalized_xy_coords:
        base_vertices_3d.append([x, y, -base_depth_param / 2.0, 1.0])

    base_faces = []
    # 1. Треугольники передней грани (обход против часовой стрелки для нормали +Z)
    for i, j, k in front_face_triangles_idxs:
        base_faces.append([i, j, k])

    # 2. Треугольники задней грани (смещаем индексы и меняем порядок для нормали -Z)
    for i, j, k in front_face_triangles_idxs:
        base_faces.append([k + num_total_front_vertices,
                           j + num_total_front_vertices,
                           i + num_total_front_vertices])

    # 3. Боковые грани для внешнего контура
    for i in range(outer_contour_vertex_count):
        p1_front = i
        p2_front = (i + 1) % outer_contour_vertex_count
        p1_back = p1_front + num_total_front_vertices
        p2_back = p2_front + num_total_front_vertices
        base_faces.append([p1_front, p1_back, p2_back])
        base_faces.append([p1_front, p2_back, p2_front])

        # 4. Боковые грани для отверстий
    current_hole_start_index_offset = outer_contour_vertex_count
    for hole_v_count in holes_vertex_counts:
        for i in range(hole_v_count):
            p1_front_hole = current_hole_start_index_offset + i
            p2_front_hole = current_hole_start_index_offset + ((i + 1) % hole_v_count)
            p1_back_hole = p1_front_hole + num_total_front_vertices
            p2_back_hole = p2_front_hole + num_total_front_vertices
            base_faces.append([p1_front_hole, p2_back_hole, p1_back_hole])
            base_faces.append([p1_front_hole, p2_front_hole, p2_back_hole])
        current_hole_start_index_offset += hole_v_count

    return base_vertices_3d, base_faces


def create_letter_V():
    """ Создает геометрию для буквы 'В' """
    v_front_xy = [
        # Внешний контур (11 вершин, индексы 0-10)
        (-0.4, 0.5),  # 0
        (-0.2, 0.5),  # 1
        (0.25, 0.5),  # 2
        (0.45, 0.3),  # 3
        (0.45, 0.05),  # 4
        (0.05, 0.0),  # 5
        (0.45, -0.05),  # 6
        (0.45, -0.3),  # 7
        (0.25, -0.5),  # 8
        (-0.2, -0.5),  # 9
        (-0.4, -0.5),  # 10

        # Внутренний контур верхнего отверстия (4 вершины, индексы 11-14)
        (-0.2, 0.3),  # 11
        (0.15, 0.3),  # 12
        (0.2, 0.1),  # 13
        (-0.2, 0.05),  # 14

        # Внутренний контур нижнего отверстия (4 вершины, индексы 15-18)
        (-0.2, -0.05),  # 15
        (0.2, -0.1),  # 16
        (0.15, -0.3),  # 17
        (-0.2, -0.3)  # 18
    ]
    outer_contour_v_count = 11
    holes_v_counts = [4, 4]

    v_front_triangles_corrected_V = [
        (0, 1, 9), (0, 9, 10),
        (1, 5, 14), (1, 14, 11),
        (5, 9, 18), (5, 18, 15),
        (5, 14, 15),
        (1, 2, 12), (1, 12, 11),
        (2, 3, 13), (2, 13, 12),
        (3, 4, 13),
        (4, 5, 14), (4, 14, 13),
        (5, 6, 16), (5, 16, 15),
        (6, 7, 17), (6, 17, 16),
        (7, 8, 17),
        (8, 9, 18), (8, 18, 17),
    ]

    base_depth_val = 0.2  # Это будет толщина Z до масштабирования Letter3D
    verts, faces = _create_extruded_geometry(v_front_xy, v_front_triangles_corrected_V,
                                             outer_contour_v_count, holes_v_counts, base_depth_val)
    return Letter3D(verts, faces, "В")


def create_letter_D():
    """ Создает геометрию для русской буквы 'Д' """
    # Координаты для "домика"
    # Y-координаты: верх=0.5, перекладина=0.0, низ=-0.5
    # X-координаты: ширина ножек ~0.2, общая ширина ~0.8

    d_front_xy = [
        # Внешний контур (8 вершин, индексы 0-7)
        (-0.4, 0.0),  # 0: левый верхний угол перекладины
        (-0.4, 0.5),  # 1: левая верхняя точка "крыши"
        (0.0, 0.8),  # 2: вершина "крыши" (выше, чем 0.5)
        (0.4, 0.5),  # 3: правая верхняя точка "крыши"
        (0.4, 0.0),  # 4: правый верхний угол перекладины
        (0.3, -0.5),  # 5: правая нижняя ножка
        (0.1, -0.5),  # 6: правая внутренняя ножка
        (-0.1, -0.5),  # 7: левая внутренняя ножка
        (-0.3, -0.5)  # 8: левая нижняя ножка
        # Вернулись к (0,1) ? нет, (0,1) уже есть.
        # Нужна еще (-0.4, 0.0) - она уже есть (0)
        # Нужно замкнуть контур.
        # (0.4, 0.0) -> (0.3, 0.0) -> (0.3, -0.5) -> (0.1, -0.5) -> (0.1, 0.0)
        # -> (-0.1, 0.0) -> (-0.1, -0.5) -> (-0.3, -0.5) -> (-0.3, 0.0) -> (-0.4,0.0)
    ]
    # Переопределяем вершины для русской "Д"
    d_front_xy = [
        # Внешний контур (10 вершин, индексы 0-9)
        (-0.4, 0.0),  # 0: левый верхний угол перекладины
        (-0.4, 0.5),  # 1: левая точка скоса крыши
        (0.0, 0.75),  # 2: пик крыши
        (0.4, 0.5),  # 3: правая точка скоса крыши
        (0.4, 0.0),  # 4: правый верхний угол перекладины
        (0.3, 0.0),  # 5: правая внешняя точка ножки (верх)
        (0.3, -0.5),  # 6: правая внешняя точка ножки (низ)
        (0.1, -0.5),  # 7: правая внутренняя точка ножки (низ)
        # Пропуск для отверстия
        (-0.1, -0.5),  # 8: левая внутренняя точка ножки (низ)
        (-0.3, -0.5),  # 9: левая внешняя точка ножки (низ)
        (-0.3, 0.0),  # 10: левая внешняя точка ножки (верх)
        # Замыкается на 0

        # Внутренний контур (отверстие под перекладиной) (4 вершины, индексы 11-14)
        # Это пространство между ножками, под основной перекладиной.
        (0.1, 0.0),  # 11: правая верхняя точка отверстия
        (0.1, -0.35),  # 12: правая нижняя точка отверстия
        (-0.1, -0.35),  # 13: левая нижняя точка отверстия
        (-0.1, 0.0),  # 14: левая верхняя точка отверстия
    ]
    outer_contour_v_count = 11  # 0-10
    holes_v_counts = [4]  # 11-14

    # Треугольники для передней грани (обход против часовой стрелки)
    d_front_triangles = [
        # "Крыша"
        (0, 1, 2), (0, 2, 4), (1, 3, 2), (3, 4, 2),  # Плоская крыша, если (0,1,3,4) - прямоугольник. Тут скосы.
        # Корректная крыша:
        (10, 0, 1), (1, 2, 3), (3, 4, 5), (10, 1, 2), (2, 3, 5),  # Не совсем.
        # Проще: (0,1,2), (0,2,10) - левый скос + часть перекладины. (4,3,2), (4,2,5) - правый скос + часть.
        # И центральный (10,2,5)
        (10, 0, 1), (10, 1, 2), (10, 2, 5),  # Левый скат и часть перемычки
        (5, 2, 3), (5, 3, 4),  # Правый скат и часть перемычки

        # Ножки и пространство над отверстием
        # Правая ножка
        (5, 6, 7), (5, 7, 11),
        # Левая ножка
        (10, 9, 8), (10, 8, 14),

        # Перекладина над отверстием
        (10, 5, 11), (10, 11, 14)
    ]

    # Важно: _create_extruded_geometry теперь сама центрирует и нормализует по высоте до 1.0.
    # Поэтому абсолютные значения XY не так критичны, как их относительные пропорции.
    base_depth_val = 0.2
    verts, faces = _create_extruded_geometry(d_front_xy, d_front_triangles,
                                             outer_contour_v_count, holes_v_counts, base_depth_val)
    return Letter3D(verts, faces, "Д")


# --- Виджет для отображения 3D ---
# (Копипаста RenderWidget из предыдущего ответа, т.к. он не менялся)
class RenderWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.objects = []  # Список Letter3D объектов
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Параметры камеры
        self.camera_eye = np.array([0.0, 0.0, 3.0])  # Позиция камеры (ближе для нормализованных букв)
        self.camera_target = np.array([0.0, 0.0, 0.0])  # Точка, куда смотрит камера
        self.camera_up = np.array([0.0, 1.0, 0.0])  # Направление "верха" для камеры
        self.fov = 60.0  # Угол обзора в градусах
        self.near_plane = 0.1
        self.far_plane = 100.0

        self.view_matrix = np.identity(4)
        self.projection_matrix = np.identity(4)
        self.update_view_projection_matrices()

        self.auto_scale_scene = True  # Автомасштабирование при первом показе

        # Для Z-буфера и шейдинга
        self.z_buffer = None
        self.frame_buffer = None  # QImage для рисования
        self.current_shading_mode = "wireframe"  # wireframe, monotone, gouraud
        self.light_position = np.array([2.0, 3.0, 4.0, 1.0])  # В мировых координатах (w=1 для точки)
        self.light_color = QColor(255, 255, 255)
        self.ambient_color = QColor(50, 50, 50)
        # Базовый цвет объектов (можно задавать индивидуально или глобально)
        self.object_base_color_V = QColor(150, 150, 200)
        self.object_base_color_D = QColor(200, 150, 150)

    def add_object(self, obj):
        self.objects.append(obj)
        # Установить начальные размеры в UI для нового объекта, если он первый
        if len(self.objects) == 1 and hasattr(self, 'parent_main_window'):
            self.parent_main_window.update_object_controls_display()
        self.update()

    def update_view_projection_matrices(self):
        aspect_ratio = self.width() / self.height() if self.height() > 0 else 1.0
        self.projection_matrix = perspective_projection_matrix(self.fov, aspect_ratio, self.near_plane, self.far_plane)
        self.view_matrix = look_at_matrix(self.camera_eye, self.camera_target, self.camera_up)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.width() > 0 and self.height() > 0:
            self.frame_buffer = QImage(self.size(), QImage.Format_RGB32)
            self.z_buffer = np.full((self.width(), self.height()), self.far_plane + 1.0, dtype=float)
            self.update_view_projection_matrices()
        self.update()

    def project_vertex(self, vertex_world_homo, viewport_w, viewport_h):
        vertex_view = self.view_matrix @ vertex_world_homo
        vertex_clip = self.projection_matrix @ vertex_view

        w_clip = vertex_clip[3]
        if abs(w_clip) < 1e-6:
            return None, float('inf'), None

        ndc_x = vertex_clip[0] / w_clip
        ndc_y = vertex_clip[1] / w_clip
        ndc_z = vertex_clip[2] / w_clip

        if not (-1.001 <= ndc_z <= 1.001):
            return None, float('inf'), None

        screen_x = (ndc_x + 1.0) * 0.5 * viewport_w
        screen_y = (1.0 - ndc_y) * 0.5 * viewport_h

        depth_value_for_z_buffer = (ndc_z + 1.0) * 0.5
        return QPointF(screen_x, screen_y), depth_value_for_z_buffer, vertex_view

    def paintEvent(self, event):
        if self.width() <= 0 or self.height() <= 0: return

        if not self.frame_buffer or self.frame_buffer.size() != self.size():
            self.frame_buffer = QImage(self.size(), QImage.Format_RGB32)
            self.z_buffer = np.full((self.width(), self.height()), self.far_plane + 1.0, dtype=float)

        self.frame_buffer.fill(Qt.black)
        self.z_buffer.fill(self.far_plane + 1.0)

        painter = QPainter(self.frame_buffer)
        painter.setRenderHint(QPainter.Antialiasing)

        viewport_w = self.width()
        viewport_h = self.height()

        if self.auto_scale_scene and self.objects:
            all_verts_world = []
            for obj in self.objects:
                all_verts_world.extend(obj.current_vertices[:, :3])

            if all_verts_world:
                all_verts_world_np = np.array(all_verts_world)
                center = np.mean(all_verts_world_np, axis=0)

                max_coord = np.max(all_verts_world_np, axis=0)
                min_coord = np.min(all_verts_world_np, axis=0)
                bb_diag = max_coord - min_coord
                # Выбираем наибольший размер из X, Y, Z для оценки масштаба сцены
                max_bb_dim = np.max(np.abs(bb_diag)) if bb_diag.size > 0 else 1.0
                max_bb_dim = max(max_bb_dim, 1.0)

                cam_dist = max_bb_dim * 1.5 / math.tan(math.radians(self.fov / 2.0))
                cam_dist = max(cam_dist, 1.5)  # Минимальное расстояние, чтобы буквы не были слишком близко

                self.camera_eye = center + np.array([0, max_bb_dim * 0.1, cam_dist])
                self.camera_target = center
                self.update_view_projection_matrices()
            self.auto_scale_scene = False

        for obj_idx, obj in enumerate(self.objects):
            transformed_vertices_cache = {}

            current_obj_base_color = self.object_base_color_V if obj.char_name == "В" else self.object_base_color_D

            if self.current_shading_mode == "wireframe":
                painter.setPen(QPen(QColor(0, 255, 0) if obj.char_name == "В" else QColor(255, 100, 100), 1))
                for edge_indices in obj.edges:
                    v1_world = obj.current_vertices[edge_indices[0]]
                    v2_world = obj.current_vertices[edge_indices[1]]

                    p1_screen, _, _ = self.project_vertex(v1_world, viewport_w, viewport_h)
                    p2_screen, _, _ = self.project_vertex(v2_world, viewport_w, viewport_h)

                    if p1_screen and p2_screen:
                        painter.drawLine(p1_screen, p2_screen)

            else:
                renderable_faces_info = []
                for face_v_indices in obj.faces:
                    screen_points = []
                    view_space_vertices_for_face = []

                    valid_face = True
                    for v_idx in face_v_indices:
                        if v_idx not in transformed_vertices_cache:
                            v_w = obj.current_vertices[v_idx]
                            p_screen, depth_ndc_01, v_view_homo = self.project_vertex(v_w, viewport_w, viewport_h)
                            transformed_vertices_cache[v_idx] = (p_screen, depth_ndc_01, v_view_homo)

                        p_screen, depth_ndc_01, v_view_homo = transformed_vertices_cache[v_idx]

                        if p_screen is None:
                            valid_face = False
                            break
                        screen_points.append(p_screen)
                        view_space_vertices_for_face.append(v_view_homo)

                    if not valid_face or len(screen_points) < 3:
                        continue

                    p0_view_h = view_space_vertices_for_face[0]
                    p1_view_h = view_space_vertices_for_face[1]
                    p2_view_h = view_space_vertices_for_face[2]

                    # Проверка на вырожденные вершины в пространстве вида (очень близкие w)
                    if abs(p0_view_h[3]) < 1e-6 or abs(p1_view_h[3]) < 1e-6 or abs(p2_view_h[3]) < 1e-6:
                        continue

                    p0_view_dec = p0_view_h[:3] / p0_view_h[3]
                    p1_view_dec = p1_view_h[:3] / p1_view_h[3]
                    p2_view_dec = p2_view_h[:3] / p2_view_h[3]

                    face_normal_view = np.cross(p1_view_dec - p0_view_dec, p2_view_dec - p0_view_dec)
                    norm_len = np.linalg.norm(face_normal_view)
                    if norm_len < 1e-6: continue
                    face_normal_view = face_normal_view / norm_len

                    view_dir_to_camera = -p0_view_dec
                    view_dir_norm = np.linalg.norm(view_dir_to_camera)
                    if view_dir_norm > 1e-6:
                        view_dir_to_camera = view_dir_to_camera / view_dir_norm
                    else:
                        view_dir_to_camera = np.array([0, 0, 1])

                    if np.dot(face_normal_view, view_dir_to_camera) <= 1e-4:  # Небольшой допуск
                        continue

                    renderable_faces_info.append({
                        'screen_points': screen_points,
                        'view_vertices_homo': view_space_vertices_for_face,
                        'face_normal_view': face_normal_view,
                        'object_base_color': current_obj_base_color
                    })

                for face_data in renderable_faces_info:
                    self.rasterize_triangle_with_zbuffer(
                        face_data['screen_points'],
                        face_data['view_vertices_homo'],
                        face_data['face_normal_view'],
                        face_data['object_base_color']  # Передаем базовый цвет конкретного объекта
                    )

        main_painter = QPainter(self)
        main_painter.drawImage(0, 0, self.frame_buffer)

        if self.current_shading_mode != "wireframe":
            light_screen_pos, _, _ = self.project_vertex(self.light_position, viewport_w, viewport_h)
            if light_screen_pos:
                main_painter.setPen(Qt.yellow)
                main_painter.setBrush(Qt.yellow)
                main_painter.drawEllipse(light_screen_pos, 4, 4)

        painter.end()
        main_painter.end()

    def rasterize_triangle_with_zbuffer(self, screen_coords_qpf_list, view_coords_homo_list, face_normal_view,
                                        object_base_color_param):
        p0_scr, p1_scr, p2_scr = screen_coords_qpf_list[0], screen_coords_qpf_list[1], screen_coords_qpf_list[2]
        v0_view_h, v1_view_h, v2_view_h = view_coords_homo_list[0], view_coords_homo_list[1], view_coords_homo_list[2]

        min_x = max(0, math.floor(min(p0_scr.x(), p1_scr.x(), p2_scr.x())))
        max_x = min(self.width() - 1, math.ceil(max(p0_scr.x(), p1_scr.x(), p2_scr.x())))
        min_y = max(0, math.floor(min(p0_scr.y(), p1_scr.y(), p2_scr.y())))
        max_y = min(self.height() - 1, math.ceil(max(p0_scr.y(), p1_scr.y(), p2_scr.y())))

        if min_x >= max_x or min_y >= max_y: return  # Вырожденный на экране

        inv_w0, inv_w1, inv_w2 = 1.0 / v0_view_h[3], 1.0 / v1_view_h[3], 1.0 / v2_view_h[3]

        v0_view_dec = v0_view_h[:3] * inv_w0  # v_view_dec = v_view_h[:3] / v_view_h[3]
        v1_view_dec = v1_view_h[:3] * inv_w1
        v2_view_dec = v2_view_h[:3] * inv_w2

        colors_at_vertices = [
            self.calculate_lighting(face_normal_view, v0_view_dec, object_base_color_param),
            self.calculate_lighting(face_normal_view, v1_view_dec, object_base_color_param),
            self.calculate_lighting(face_normal_view, v2_view_dec, object_base_color_param)
        ]

        vec_p0p1_scr = p1_scr - p0_scr
        vec_p0p2_scr = p2_scr - p0_scr
        area_total_twice_scr = vec_p0p1_scr.x() * vec_p0p2_scr.y() - vec_p0p1_scr.y() * vec_p0p2_scr.x()
        if abs(area_total_twice_scr) < 1e-7: return

        for x_scan in range(min_x, max_x + 1):
            for y_scan in range(min_y, max_y + 1):
                vec_p_scan_p0_scr = QPointF(x_scan + 0.5, y_scan + 0.5) - p0_scr  # Центр пикселя

                v_bc = (
                                   vec_p_scan_p0_scr.x() * vec_p0p1_scr.y() - vec_p_scan_p0_scr.y() * vec_p0p1_scr.x()) / area_total_twice_scr
                u_bc = (
                                   vec_p0p2_scr.x() * vec_p_scan_p0_scr.y() - vec_p0p2_scr.y() * vec_p_scan_p0_scr.x()) / area_total_twice_scr
                w_bc = 1.0 - u_bc - v_bc

                if w_bc >= -1e-4 and u_bc >= -1e-4 and v_bc >= -1e-4:  # Допуск для ребер
                    one_over_w_view_interp = w_bc * inv_w0 + u_bc * inv_w1 + v_bc * inv_w2
                    if abs(one_over_w_view_interp) < 1e-9: continue

                    z_ndc_interp = (w_bc * (v0_view_h[2] * inv_w0) +
                                    u_bc * (v1_view_h[2] * inv_w1) +
                                    v_bc * (v2_view_h[2] * inv_w2)) / one_over_w_view_interp

                    depth_for_z_buffer = (z_ndc_interp + 1.0) * 0.5

                    if not (0.0 <= depth_for_z_buffer <= 1.001): continue  # Допуск для дальней плоскости

                    # Округляем x_scan, y_scan до целых, если они стали float из-за +0.5
                    ix, iy = int(x_scan), int(y_scan)
                    if ix < 0 or ix >= self.width() or iy < 0 or iy >= self.height(): continue

                    if depth_for_z_buffer < self.z_buffer[ix, iy]:
                        self.z_buffer[ix, iy] = depth_for_z_buffer

                        pixel_color_final = QColor(Qt.black)
                        if self.current_shading_mode == "monotone":
                            pixel_color_final = colors_at_vertices[0]
                        elif self.current_shading_mode == "gouraud":
                            r_i = (w_bc * (colors_at_vertices[0].redF() * inv_w0) + \
                                   u_bc * (colors_at_vertices[1].redF() * inv_w1) + \
                                   v_bc * (colors_at_vertices[2].redF() * inv_w2)) / one_over_w_view_interp
                            g_i = (w_bc * (colors_at_vertices[0].greenF() * inv_w0) + \
                                   u_bc * (colors_at_vertices[1].greenF() * inv_w1) + \
                                   v_bc * (colors_at_vertices[2].greenF() * inv_w2)) / one_over_w_view_interp
                            b_i = (w_bc * (colors_at_vertices[0].blueF() * inv_w0) + \
                                   u_bc * (colors_at_vertices[1].blueF() * inv_w1) + \
                                   v_bc * (colors_at_vertices[2].blueF() * inv_w2)) / one_over_w_view_interp
                            pixel_color_final.setRgbF(min(1, max(0, r_i)), min(1, max(0, g_i)), min(1, max(0, b_i)))

                        self.frame_buffer.setPixelColor(ix, iy, pixel_color_final)

    def calculate_lighting(self, normal_view_unit, vertex_pos_view_dec, base_obj_color_param):
        light_pos_view_homo = self.view_matrix @ self.light_position
        if abs(light_pos_view_homo[3]) < 1e-6: return self.ambient_color  # Свет в бесконечности или ошибка
        light_pos_view_dec = light_pos_view_homo[:3] / light_pos_view_homo[3]

        light_dir_to_source_view = light_pos_view_dec - vertex_pos_view_dec
        norm_light_dir = np.linalg.norm(light_dir_to_source_view)
        if norm_light_dir > 1e-6:
            light_dir_to_source_view = light_dir_to_source_view / norm_light_dir

        diffuse_intensity = max(0, np.dot(normal_view_unit, light_dir_to_source_view))

        r_f = self.ambient_color.redF() + base_obj_color_param.redF() * diffuse_intensity * self.light_color.redF()
        g_f = self.ambient_color.greenF() + base_obj_color_param.greenF() * diffuse_intensity * self.light_color.greenF()
        b_f = self.ambient_color.blueF() + base_obj_color_param.blueF() * diffuse_intensity * self.light_color.blueF()

        return QColor.fromRgbF(min(1, max(0, r_f)), min(1, max(0, g_f)), min(1, max(0, b_f)))

    def set_shading_mode(self, mode_text):
        self.current_shading_mode = mode_text
        self.update()

    def set_light_pos(self, x, y, z):
        self.light_position = np.array([x, y, z, 1.0])
        self.update()


# --- Главное окно приложения ---
# (Копипаста MainWindow из предыдущего ответа, т.к. он не менялся сильно,
# кроме того, что RenderWidget теперь может ссылаться на него для обновления UI)
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторная работа №2 - 3D Объекты (Русская Д)")
        self.setGeometry(100, 100, 1200, 800)

        self.render_widget = RenderWidget()
        self.render_widget.parent_main_window = self  # Даем RenderWidget ссылку на MainWindow

        self.letter_v = create_letter_V()
        self.letter_d = create_letter_D()

        # Начальные позиции можно настроить после того, как буквы созданы и их base_model_width известен
        # Например, сдвинуть букву Д немного правее, чтобы они не перекрывались в начале
        total_width_estimate = self.letter_v.base_model_width + self.letter_d.base_model_width
        self.letter_v.set_translation(-self.letter_v.base_model_width * 0.6, 0, 0)
        self.letter_d.set_translation(self.letter_d.base_model_width * 0.6, 0, 0)

        self.render_widget.add_object(self.letter_v)
        self.render_widget.add_object(self.letter_d)
        self.selected_object_idx = 0

        self.setup_ui()
        self.connect_controls()

        self.update_object_controls_display()  # Инициализировать UI значениями первого объекта

    def setup_ui(self):
        main_layout = QHBoxLayout()

        controls_panel_widget = QWidget()
        controls_panel_widget.setMaximumWidth(380)
        controls_layout = QVBoxLayout(controls_panel_widget)

        self.obj_select_combo = QComboBox()
        self.obj_select_combo.addItems([obj.char_name for obj in self.render_widget.objects])
        controls_layout.addWidget(QLabel("Выбранный объект:"))
        controls_layout.addWidget(self.obj_select_combo)

        obj_params_group = QGroupBox("Параметры объекта")
        obj_params_grid = QVBoxLayout(obj_params_group)

        param_labels = ["Высота (Y):", "Ширина (X):", "Глубина (Z):"]
        self.obj_dim_spins = []
        for i, label_text in enumerate(param_labels):
            row_layout = QHBoxLayout()
            row_layout.addWidget(QLabel(label_text))
            spin = QDoubleSpinBox()
            spin.setRange(0.05, 10.0);
            spin.setSingleStep(0.05)  # Меньший шаг для толщины
            self.obj_dim_spins.append(spin)
            row_layout.addWidget(spin)
            obj_params_grid.addLayout(row_layout)
        controls_layout.addWidget(obj_params_group)

        obj_transform_group = QGroupBox("Трансформации объекта")
        obj_transform_grid = QVBoxLayout(obj_transform_group)
        transform_labels = ["Позиция X:", "Позиция Y:", "Позиция Z:",
                            "Вращение X:", "Вращение Y:", "Вращение Z:"]
        self.obj_transform_spins = []
        ranges = [(-10, 10)] * 3 + [(-360, 360)] * 3
        steps = [0.1] * 3 + [5] * 3
        for i, label_text in enumerate(transform_labels):
            row_layout = QHBoxLayout()
            row_layout.addWidget(QLabel(label_text))
            spin = QDoubleSpinBox()
            spin.setRange(ranges[i][0], ranges[i][1]);
            spin.setSingleStep(steps[i])
            self.obj_transform_spins.append(spin)
            row_layout.addWidget(spin)
            obj_transform_grid.addLayout(row_layout)
        controls_layout.addWidget(obj_transform_group)

        cam_group = QGroupBox("Управление камерой")
        cam_grid = QVBoxLayout(cam_group)
        cam_labels = ["Камера X:", "Камера Y:", "Камера Z:", "Цель X:", "Цель Y:", "Цель Z:", "FOV:"]
        self.cam_spins = []
        cam_ranges_ = [(-20, 20)] * 6 + [(10, 120)]
        cam_steps_ = [0.5] * 6 + [5]
        cam_defaults_ = [self.render_widget.camera_eye[0], self.render_widget.camera_eye[1],
                         self.render_widget.camera_eye[2],
                         self.render_widget.camera_target[0], self.render_widget.camera_target[1],
                         self.render_widget.camera_target[2],
                         self.render_widget.fov]
        for i, label_text in enumerate(cam_labels):
            row_layout = QHBoxLayout()
            row_layout.addWidget(QLabel(label_text))
            spin = QDoubleSpinBox()
            spin.setRange(cam_ranges_[i][0], cam_ranges_[i][1]);
            spin.setSingleStep(cam_steps_[i]);
            spin.setValue(cam_defaults_[i])
            self.cam_spins.append(spin)
            row_layout.addWidget(spin)
            cam_grid.addLayout(row_layout)
        self.reset_cam_button = QPushButton("Сброс камеры (Автомасштаб)")
        cam_grid.addWidget(self.reset_cam_button)
        controls_layout.addWidget(cam_group)

        render_group = QGroupBox("Настройки рендеринга")
        render_grid = QVBoxLayout(render_group)
        render_grid.addWidget(QLabel("Режим закраски:"))
        self.shading_combo = QComboBox()
        self.shading_combo.addItems(["wireframe", "monotone", "gouraud"])
        render_grid.addWidget(self.shading_combo)
        light_labels_ = ["Свет X:", "Свет Y:", "Свет Z:"]
        self.light_spins = []
        light_defaults_ = self.render_widget.light_position[:3]
        for i, label_text in enumerate(light_labels_):
            row_layout = QHBoxLayout()
            row_layout.addWidget(QLabel(label_text))
            spin = QDoubleSpinBox()
            spin.setRange(-20, 20);
            spin.setSingleStep(0.5);
            spin.setValue(light_defaults_[i])
            self.light_spins.append(spin)
            row_layout.addWidget(spin)
            render_grid.addLayout(row_layout)
        controls_layout.addWidget(render_group)

        controls_layout.addStretch()
        main_widget = QWidget()
        main_layout.addWidget(self.render_widget, 1)
        main_layout.addWidget(controls_panel_widget)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def connect_controls(self):
        self.obj_select_combo.currentIndexChanged.connect(self.on_selected_object_changed)
        for spin in self.obj_dim_spins: spin.valueChanged.connect(self.on_object_params_or_transform_changed)
        for spin in self.obj_transform_spins: spin.valueChanged.connect(self.on_object_params_or_transform_changed)
        for spin in self.cam_spins: spin.valueChanged.connect(self.on_camera_params_changed)
        for spin in self.light_spins: spin.valueChanged.connect(self.on_light_params_changed)
        self.reset_cam_button.clicked.connect(self.reset_camera_view)
        self.shading_combo.currentTextChanged.connect(self.render_widget.set_shading_mode)

    def on_selected_object_changed(self, index):
        self.selected_object_idx = index
        self.update_object_controls_display()

    def update_object_controls_display(self):
        if not self.render_widget.objects or self.selected_object_idx >= len(self.render_widget.objects): return
        obj = self.render_widget.objects[self.selected_object_idx]

        dims_to_set = [obj.height, obj.width, obj.depth]
        for i, spin in enumerate(self.obj_dim_spins):
            spin.blockSignals(True);
            spin.setValue(dims_to_set[i]);
            spin.blockSignals(False)

        transforms_to_set = [
            obj.translation[0], obj.translation[1], obj.translation[2],
            obj.rotation_angles[0], obj.rotation_angles[1], obj.rotation_angles[2]
        ]
        for i, spin in enumerate(self.obj_transform_spins):
            spin.blockSignals(True);
            spin.setValue(transforms_to_set[i]);
            spin.blockSignals(False)
        # self.render_widget.update() # Не нужно, т.к. это только обновление UI

    def on_object_params_or_transform_changed(self):  # Объединенный обработчик
        if not self.render_widget.objects or self.selected_object_idx >= len(self.render_widget.objects): return
        obj = self.render_widget.objects[self.selected_object_idx]

        new_h = self.obj_dim_spins[0].value()
        new_w = self.obj_dim_spins[1].value()
        new_d = self.obj_dim_spins[2].value()
        obj.set_dimensions(new_h, new_w, new_d)

        obj.set_translation(self.obj_transform_spins[0].value(),
                            self.obj_transform_spins[1].value(),
                            self.obj_transform_spins[2].value())
        obj.set_rotation(self.obj_transform_spins[3].value(),
                         self.obj_transform_spins[4].value(),
                         self.obj_transform_spins[5].value())
        self.render_widget.update()

    def on_camera_params_changed(self):
        self.render_widget.camera_eye = np.array(
            [self.cam_spins[0].value(), self.cam_spins[1].value(), self.cam_spins[2].value()])
        self.render_widget.camera_target = np.array(
            [self.cam_spins[3].value(), self.cam_spins[4].value(), self.cam_spins[5].value()])
        self.render_widget.fov = self.cam_spins[6].value()
        self.render_widget.update_view_projection_matrices()
        self.render_widget.update()

    def reset_camera_view(self):
        self.render_widget.auto_scale_scene = True
        self.render_widget.update()
        cam_defaults_updated = [self.render_widget.camera_eye[0], self.render_widget.camera_eye[1],
                                self.render_widget.camera_eye[2],
                                self.render_widget.camera_target[0], self.render_widget.camera_target[1],
                                self.render_widget.camera_target[2],
                                self.render_widget.fov]
        for i, spin in enumerate(self.cam_spins):
            spin.blockSignals(True);
            spin.setValue(cam_defaults_updated[i]);
            spin.blockSignals(False)

    def on_light_params_changed(self):
        self.render_widget.set_light_pos(
            self.light_spins[0].value(),
            self.light_spins[1].value(),
            self.light_spins[2].value()
        )

    def keyPressEvent(self, event):
        step = 0.1 * (self.render_widget.camera_eye[2] / 3.0)  # Масштабируем шаг от расстояния камеры
        step = max(0.05, step)  # Минимальный шаг
        angle_step = 2.0
        cam_changed = False
        rw = self.render_widget

        forward_vec = rw.camera_target - rw.camera_eye
        if np.linalg.norm(forward_vec) > 1e-6: forward_vec /= np.linalg.norm(forward_vec)
        right_vec = np.cross(forward_vec, rw.camera_up)
        if np.linalg.norm(right_vec) > 1e-6: right_vec /= np.linalg.norm(right_vec)

        true_up_vec = np.cross(right_vec, forward_vec)  # Для движения вверх/вниз по оси камеры

        key = event.key()
        if key == Qt.Key_W:
            rw.camera_eye += forward_vec * step;
            rw.camera_target += forward_vec * step;
            cam_changed = True
        elif key == Qt.Key_S:
            rw.camera_eye -= forward_vec * step;
            rw.camera_target -= forward_vec * step;
            cam_changed = True
        elif key == Qt.Key_A:
            rw.camera_eye -= right_vec * step;
            rw.camera_target -= right_vec * step;
            cam_changed = True
        elif key == Qt.Key_D:
            rw.camera_eye += right_vec * step;
            rw.camera_target += right_vec * step;
            cam_changed = True
        elif key == Qt.Key_R:  # Вверх по оси камеры
            rw.camera_eye += true_up_vec * step;
            rw.camera_target += true_up_vec * step;
            cam_changed = True
        elif key == Qt.Key_F:  # Вниз по оси камеры
            rw.camera_eye -= true_up_vec * step;
            rw.camera_target -= true_up_vec * step;
            cam_changed = True
        elif key == Qt.Key_Q:  # Поворот влево (рыскание)
            rot_mat = rotation_matrix_y(math.radians(angle_step))  # Вокруг мирового Y
            # Вращаем target относительно eye
            target_rel_eye = rw.camera_target - rw.camera_eye
            new_target_rel_eye = (rot_mat @ np.append(target_rel_eye, 1))[:3]
            rw.camera_target = rw.camera_eye + new_target_rel_eye
            cam_changed = True
        elif key == Qt.Key_E:  # Поворот вправо (рыскание)
            rot_mat = rotation_matrix_y(math.radians(-angle_step))  # Вокруг мирового Y
            target_rel_eye = rw.camera_target - rw.camera_eye
            new_target_rel_eye = (rot_mat @ np.append(target_rel_eye, 1))[:3]
            rw.camera_target = rw.camera_eye + new_target_rel_eye
            cam_changed = True
        # Тангаж (pitch) - поворот вокруг оси "right_vec" камеры
        elif key == Qt.Key_T:  # Вверх
            rot_mat_pitch = self.create_rotation_around_vector(right_vec, math.radians(angle_step))
            target_rel_eye = rw.camera_target - rw.camera_eye
            new_target_rel_eye = (rot_mat_pitch @ np.append(target_rel_eye, 1))[:3]
            rw.camera_target = rw.camera_eye + new_target_rel_eye
            # Также нужно повернуть вектор "up" камеры, чтобы избежать крена
            rw.camera_up = (rot_mat_pitch @ np.append(rw.camera_up, 1))[:3]
            cam_changed = True
        elif key == Qt.Key_G:  # Вниз
            rot_mat_pitch = self.create_rotation_around_vector(right_vec, math.radians(-angle_step))
            target_rel_eye = rw.camera_target - rw.camera_eye
            new_target_rel_eye = (rot_mat_pitch @ np.append(target_rel_eye, 1))[:3]
            rw.camera_target = rw.camera_eye + new_target_rel_eye
            rw.camera_up = (rot_mat_pitch @ np.append(rw.camera_up, 1))[:3]
            cam_changed = True

        if cam_changed:
            self.render_widget.update_view_projection_matrices()
            cam_vals_ui = [rw.camera_eye[0], rw.camera_eye[1], rw.camera_eye[2],
                           rw.camera_target[0], rw.camera_target[1], rw.camera_target[2], rw.fov]
            for i, spin in enumerate(self.cam_spins):
                spin.blockSignals(True);
                spin.setValue(cam_vals_ui[i]);
                spin.blockSignals(False)
            self.render_widget.update()

        super().keyPressEvent(event)

    def create_rotation_around_vector(self, axis, angle_rad):
        # Формула вращения Родригеса для создания матрицы поворота вокруг произвольной оси
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis[0], axis[1], axis[2]
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        C = 1 - c
        return np.array([
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s, 0],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s, 0],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c, 0],
            [0, 0, 0, 1]
        ])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())