import sys
import math
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QDoubleSpinBox, QPushButton, QFrame, QSizePolicy, QTabWidget
)
from PySide6.QtGui import QPainter, QColor, QPen, QImage, QPolygonF, QBrush, QFont
from PySide6.QtCore import Qt, QPointF, QTimer, QSize


# ==============================================================================
# 3D Математика (Векторы и Матрицы) - [КОД ОСТАЕТСЯ ПРЕЖНИМ]
# ==============================================================================
class Vec3:
    # Класс для представления 3D вектора или точки
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot(self, other):
        # Скалярное произведение
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        # Векторное произведение
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def length(self):
        # Длина вектора
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalized(self):
        # Нормализованный вектор
        l = self.length()
        if l == 0:
            return Vec3(0, 0, 0)
        return Vec3(self.x / l, self.y / l, self.z / l)

    def to_vec4(self, w=1.0):
        # Преобразование в Vec4 (для гомогенных координат)
        return Vec4(self.x, self.y, self.z, w)

    def __repr__(self):
        return f"Vec3({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"


class Vec4:
    # Класс для представления 4D вектора (гомогенные координаты)
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)

    def perspective_divide(self):
        # Перспективное деление
        if self.w == 0 or self.w == 1.0:  # Избегаем деления на 0 или если уже в декартовых
            return Vec3(self.x, self.y, self.z)
        return Vec3(self.x / self.w, self.y / self.w, self.z / self.w)

    def __repr__(self):
        return f"Vec4({self.x:.2f}, {self.y:.2f}, {self.z:.2f}, {self.w:.2f})"


class Mat4:
    # Класс для представления матрицы 4x4
    def __init__(self):
        # Инициализация единичной матрицей
        self.m = [[0.0] * 4 for _ in range(4)]
        for i in range(4):
            self.m[i][i] = 1.0

    @staticmethod
    def identity():
        return Mat4()

    @staticmethod
    def translation(tx, ty, tz):
        # Матрица трансляции
        mat = Mat4.identity()
        mat.m[0][3] = tx
        mat.m[1][3] = ty
        mat.m[2][3] = tz
        return mat

    @staticmethod
    def scaling(sx, sy, sz):
        # Матрица масштабирования
        mat = Mat4.identity()
        mat.m[0][0] = sx
        mat.m[1][1] = sy
        mat.m[2][2] = sz
        return mat

    @staticmethod
    def rotation_x(angle_rad):
        # Матрица поворота вокруг оси X
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        mat = Mat4.identity()
        mat.m[1][1] = c
        mat.m[1][2] = -s
        mat.m[2][1] = s
        mat.m[2][2] = c
        return mat

    @staticmethod
    def rotation_y(angle_rad):
        # Матрица поворота вокруг оси Y
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        mat = Mat4.identity()
        mat.m[0][0] = c
        mat.m[0][2] = s
        mat.m[2][0] = -s
        mat.m[2][2] = c
        return mat

    @staticmethod
    def rotation_z(angle_rad):
        # Матрица поворота вокруг оси Z
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        mat = Mat4.identity()
        mat.m[0][0] = c
        mat.m[0][1] = -s
        mat.m[1][0] = s
        mat.m[1][1] = c
        return mat

    def __mul__(self, other):
        # Умножение матриц или матрицы на вектор
        if isinstance(other, Mat4):
            res = Mat4()
            res.m = [[0.0] * 4 for _ in range(4)]  # Обнуляем результат
            for i in range(4):
                for j in range(4):
                    sum_val = 0.0
                    for k in range(4):
                        sum_val += self.m[i][k] * other.m[k][j]
                    res.m[i][j] = sum_val
            return res
        elif isinstance(other, Vec4):
            res_v = [0.0] * 4
            for i in range(4):
                sum_val = 0.0
                for j in range(4):
                    sum_val += self.m[i][j] * [other.x, other.y, other.z, other.w][j]
                res_v[i] = sum_val
            return Vec4(res_v[0], res_v[1], res_v[2], res_v[3])
        return NotImplemented

    @staticmethod
    def perspective(fov_y_rad, aspect_ratio, z_near, z_far):
        # Матрица перспективного проецирования
        mat = Mat4()
        mat.m = [[0.0] * 4 for _ in range(4)]  # Обнуляем

        tan_half_fovy = math.tan(fov_y_rad / 2.0)

        mat.m[0][0] = 1.0 / (aspect_ratio * tan_half_fovy)
        mat.m[1][1] = 1.0 / (tan_half_fovy)
        mat.m[2][2] = -(z_far + z_near) / (z_far - z_near)
        mat.m[2][3] = -(2.0 * z_far * z_near) / (z_far - z_near)
        mat.m[3][2] = -1.0
        return mat

    @staticmethod
    def look_at(eye: Vec3, target: Vec3, up: Vec3):
        # Матрица вида (камеры)
        f = (target - eye).normalized()
        s_vec = f.cross(up).normalized()  # Renamed to avoid conflict with math.sin
        u = s_vec.cross(f)

        mat = Mat4.identity()
        mat.m[0][0] = s_vec.x
        mat.m[0][1] = s_vec.y
        mat.m[0][2] = s_vec.z
        mat.m[1][0] = u.x
        mat.m[1][1] = u.y
        mat.m[1][2] = u.z
        mat.m[2][0] = -f.x
        mat.m[2][1] = -f.y
        mat.m[2][2] = -f.z

        mat.m[0][3] = -s_vec.dot(eye)
        mat.m[1][3] = -u.dot(eye)
        mat.m[2][3] = f.dot(eye)
        return mat


# ==============================================================================
# Геометрия и Сцена
# ==============================================================================

class Mesh:
    # Класс для хранения геометрии объекта (вершины и грани)
    def __init__(self, vertices=None, faces=None, color=QColor("gray")):
        self.vertices = vertices if vertices else []  # Список Vec3
        self.faces = faces if faces else []  # Список списков индексов вершин
        self.color = color  # Базовый цвет меша


def _extrude_polygon(polygon_2d, depth, color):
    # Вспомогательная функция для создания 3D меша путем экструзии 2D полигона
    vertices = []
    faces = []

    num_poly_verts = len(polygon_2d)
    if num_poly_verts < 3:
        return Mesh([], [], color)

    # Передняя грань
    front_face_indices = []
    for i in range(num_poly_verts):
        vertices.append(Vec3(polygon_2d[i].x, polygon_2d[i].y, -depth / 2.0))
        front_face_indices.append(i)
    faces.append(front_face_indices)

    # Задняя грань
    back_face_indices = []
    for i in range(num_poly_verts):
        vertices.append(Vec3(polygon_2d[i].x, polygon_2d[i].y, depth / 2.0))
        back_face_indices.append(num_poly_verts + i)
    faces.append(back_face_indices[::-1])  # Ensure correct winding for back face

    # Боковые грани
    for i in range(num_poly_verts):
        idx0_front = i
        idx1_front = (i + 1) % num_poly_verts
        idx0_back = num_poly_verts + i
        idx1_back = num_poly_verts + ((i + 1) % num_poly_verts)

        faces.append([idx0_front, idx1_front, idx1_back, idx0_back])  # CCW when viewed from outside

    return Mesh(vertices, faces, color)


def create_letter_c_mesh(height, stroke_width, depth, segments=20):
    # Создание меша для буквы "C" путем объединения экструдированных сегментов
    outer_radius = height / 2.0
    inner_radius = outer_radius - stroke_width
    if inner_radius <= 0:
        inner_radius = max(0.01, stroke_width * 0.1)  # Ensure positive and non-zero radius

    angle_start_rad = math.pi / 5
    angle_end_rad = 2 * math.pi - angle_start_rad

    letter_color = QColor("darkCyan")

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    # Вспомогательная функция для добавления компонента (сегмента)
    def add_component_local(poly_2d_local):
        nonlocal vertex_offset, all_vertices, all_faces
        if not poly_2d_local or len(poly_2d_local) < 3:
            return

        mesh_part = _extrude_polygon(poly_2d_local, depth, letter_color)
        if not mesh_part.vertices:
            return

        for v_part in mesh_part.vertices:
            all_vertices.append(v_part)

        for f_indices_part in mesh_part.faces:
            all_faces.append([idx + vertex_offset for idx in f_indices_part])
        vertex_offset += len(mesh_part.vertices)

    # Конец вспомогательной функции

    if segments <= 0: segments = 1  # Ensure at least one segment
    delta_angle = (angle_end_rad - angle_start_rad) / segments

    for i in range(segments):
        current_angle = angle_start_rad + i * delta_angle
        next_angle = angle_start_rad + (i + 1) * delta_angle

        # Точки для текущего четырехугольного сегмента
        # Внешние точки
        o1 = Vec3(outer_radius * math.cos(current_angle), outer_radius * math.sin(current_angle), 0)
        o2 = Vec3(outer_radius * math.cos(next_angle), outer_radius * math.sin(next_angle), 0)
        # Внутренние точки
        i1 = Vec3(inner_radius * math.cos(current_angle), inner_radius * math.sin(current_angle), 0)
        i2 = Vec3(inner_radius * math.cos(next_angle), inner_radius * math.sin(next_angle), 0)

        # Определяем 2D полигон для этого сегмента (четырехугольник)
        # Порядок вершин o1, o2, i2, i1 обеспечивает ориентацию против часовой стрелки (CCW) для передней грани
        segment_poly_2d = [o1, o2, i2, i1]

        add_component_local(segment_poly_2d)

    if not all_vertices and segments > 0:  # Handle cases like zero stroke width leading to no geometry
        # Fallback to create a minimal (possibly invisible or tiny) mesh to avoid errors downstream
        # This might happen if inner_radius becomes equal or very close to outer_radius.
        # print(f"Warning: Letter C resulted in no vertices. Params: H={height}, SW={stroke_width}, D={depth}, Segs={segments}")
        # Create a tiny placeholder polygon if needed, e.g., a small square
        placeholder_poly = [Vec3(-0.01, -0.01, 0), Vec3(0.01, -0.01, 0), Vec3(0.01, 0.01, 0), Vec3(-0.01, 0.01, 0)]
        add_component_local(placeholder_poly)

    return Mesh(all_vertices, all_faces, letter_color)


def create_letter_d_mesh(width, height, depth):
    # Создание меша для буквы "Д" (стандартная форма, с ножками-ступнями)
    letter_color = QColor("darkRed")

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    # --- Расчет толщины обводки (st) и высоты ступней (fh) ---
    stroke_thickness_factor = 0.20  # 20% от меньшей стороны для толщины обводки
    min_absolute_stroke_thickness = 0.05  # Минимальная абсолютная толщина

    # Начальный расчет толщины обводки
    st = min(width, height) * stroke_thickness_factor
    st = max(st, min_absolute_stroke_thickness)

    # Защита от слишком большой толщины обводки
    if 3.0 * st >= height:
        st = height / 3.0 - 0.001
    if 2.0 * st >= width:
        st = width / 2.0 - 0.001
    st = max(st, 0.001)
    fh = st

    w_half = width / 2.0
    h_half = height / 2.0
    epsilon = 1e-5

    def add_component(poly_2d):
        nonlocal vertex_offset, all_vertices, all_faces
        if not poly_2d: return
        min_x_poly = min(v.x for v in poly_2d);
        max_x_poly = max(v.x for v in poly_2d)
        min_y_poly = min(v.y for v in poly_2d);
        max_y_poly = max(v.y for v in poly_2d)
        if abs(max_x_poly - min_x_poly) < epsilon or abs(max_y_poly - min_y_poly) < epsilon: return

        mesh_part = _extrude_polygon(poly_2d, depth, letter_color)
        if not mesh_part.vertices: return
        for v_part in mesh_part.vertices: all_vertices.append(v_part)
        for f_indices_part in mesh_part.faces:
            all_faces.append([idx + vertex_offset for idx in f_indices_part])
        vertex_offset += len(mesh_part.vertices)

    y_top_bar_top_coord = h_half
    y_top_bar_bottom_coord = h_half - st
    if y_top_bar_top_coord > y_top_bar_bottom_coord + epsilon:
        add_component([Vec3(-w_half, y_top_bar_top_coord, 0), Vec3(w_half, y_top_bar_top_coord, 0),
                       Vec3(w_half, y_top_bar_bottom_coord, 0), Vec3(-w_half, y_top_bar_bottom_coord, 0)])

    y_bottom_crossbar_top_coord = -h_half + fh + st
    y_bottom_crossbar_bottom_coord = -h_half + fh
    if y_bottom_crossbar_top_coord > y_bottom_crossbar_bottom_coord + epsilon:
        add_component([Vec3(-w_half, y_bottom_crossbar_top_coord, 0), Vec3(w_half, y_bottom_crossbar_top_coord, 0),
                       Vec3(w_half, y_bottom_crossbar_bottom_coord, 0),
                       Vec3(-w_half, y_bottom_crossbar_bottom_coord, 0)])

    y_legs_top_coord = y_top_bar_bottom_coord
    y_legs_bottom_coord = y_bottom_crossbar_top_coord
    if y_legs_top_coord > y_legs_bottom_coord + epsilon:
        x_left_leg_left_coord = -w_half;
        x_left_leg_right_coord = -w_half + st
        if x_left_leg_right_coord > x_left_leg_left_coord + epsilon:
            add_component(
                [Vec3(x_left_leg_left_coord, y_legs_top_coord, 0), Vec3(x_left_leg_right_coord, y_legs_top_coord, 0),
                 Vec3(x_left_leg_right_coord, y_legs_bottom_coord, 0),
                 Vec3(x_left_leg_left_coord, y_legs_bottom_coord, 0)])
        x_right_leg_left_coord = w_half - st;
        x_right_leg_right_coord = w_half
        if x_right_leg_right_coord > x_right_leg_left_coord + epsilon:
            add_component(
                [Vec3(x_right_leg_left_coord, y_legs_top_coord, 0), Vec3(x_right_leg_right_coord, y_legs_top_coord, 0),
                 Vec3(x_right_leg_right_coord, y_legs_bottom_coord, 0),
                 Vec3(x_right_leg_left_coord, y_legs_bottom_coord, 0)])

    y_foot_top_coord = y_bottom_crossbar_bottom_coord
    y_foot_bottom_coord = -h_half
    if y_foot_top_coord > y_foot_bottom_coord + epsilon:
        x_left_foot_left_coord = -w_half;
        x_left_foot_right_coord = -w_half + st
        if x_left_foot_right_coord > x_left_foot_left_coord + epsilon:
            add_component(
                [Vec3(x_left_foot_left_coord, y_foot_top_coord, 0), Vec3(x_left_foot_right_coord, y_foot_top_coord, 0),
                 Vec3(x_left_foot_right_coord, y_foot_bottom_coord, 0),
                 Vec3(x_left_foot_left_coord, y_foot_bottom_coord, 0)])
        x_right_foot_left_coord = w_half - st;
        x_right_foot_right_coord = w_half
        if x_right_foot_right_coord > x_right_foot_left_coord + epsilon:
            add_component([Vec3(x_right_foot_left_coord, y_foot_top_coord, 0),
                           Vec3(x_right_foot_right_coord, y_foot_top_coord, 0),
                           Vec3(x_right_foot_right_coord, y_foot_bottom_coord, 0),
                           Vec3(x_right_foot_left_coord, y_foot_bottom_coord, 0)])

    return Mesh(all_vertices, all_faces, letter_color)


class SceneObject:
    # Класс для объекта на сцене (меш + трансформация)
    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.transform = Mat4.identity()  # Model matrix
        self.rotation = Vec3(0, 0, 0)  # Углы поворота в радианах (локальные)
        self.translation = Vec3(0, 0, 0)  # Положение
        self.scale = Vec3(1, 1, 1)  # Масштаб

    def update_transform(self):
        # Обновление матрицы модели на основе rotation, translation, scale
        s_mat = Mat4.scaling(self.scale.x, self.scale.y, self.scale.z)
        rx = Mat4.rotation_x(self.rotation.x)
        ry = Mat4.rotation_y(self.rotation.y)
        rz = Mat4.rotation_z(self.rotation.z)
        r = rz * ry * rx  # Standard order for Euler angles (e.g., Tait-Bryan ZYX)
        t = Mat4.translation(self.translation.x, self.translation.y, self.translation.z)
        self.transform = t * r * s_mat


class Camera:
    # Класс камеры
    def __init__(self):
        self.eye = Vec3(0, 0, 5)
        self.target = Vec3(0, 0, 0)
        self.up = Vec3(0, 1, 0)
        self.fov_y_rad = math.radians(60.0)
        self.z_near = 0.1
        self.z_far = 100.0
        self.view_matrix = Mat4.identity()
        self.projection_matrix = Mat4.identity()
        self.update_view_matrix()

    def update_view_matrix(self):
        self.view_matrix = Mat4.look_at(self.eye, self.target, self.up)

    def update_projection_matrix(self, aspect_ratio):
        if aspect_ratio <= 0: aspect_ratio = 1.0  # Guard
        self.projection_matrix = Mat4.perspective(self.fov_y_rad, aspect_ratio, self.z_near, self.z_far)

    def move(self, dx, dy, dz):  # World axis move
        offset = Vec3(dx, dy, dz)
        self.eye = self.eye + offset
        self.target = self.target + offset  # Pan camera
        self.update_view_matrix()

    def rotate_orbit(self, angle_x_rad, angle_y_rad):  # angle_x for pitch, angle_y for yaw
        dir_to_target = self.eye - self.target
        # Yaw (around world Y axis, effectively)
        # Matches standard rotation: x' = x*c - z*s, z' = x*s + z*c for CCW rotation in XZ plane
        new_x_yaw = dir_to_target.x * math.cos(angle_y_rad) - dir_to_target.z * math.sin(angle_y_rad)
        new_z_yaw = dir_to_target.x * math.sin(angle_y_rad) + dir_to_target.z * math.cos(angle_y_rad)
        dir_to_target.x = new_x_yaw
        dir_to_target.z = new_z_yaw
        # Pitch (around horizontal axis perpendicular to view direction and Up) - simplified for now
        # For simplicity, if angle_x_rad is non-zero, it would rotate around an axis s = (target-eye).cross(up)
        # This example primarily uses yaw (angle_y_rad) based on UI
        self.eye = self.target + dir_to_target
        self.update_view_matrix()

    def zoom(self, factor):
        direction = (self.eye - self.target).normalized()
        distance = (self.eye - self.target).length()
        new_distance = max(self.z_near * 1.5, distance * factor)  # Ensure not too close
        self.eye = self.target + direction * new_distance
        self.update_view_matrix()


# ==============================================================================
# Виджет Рендеринга
# ==============================================================================
class RenderWidget(QWidget):
    def __init__(self, scene_objects, camera, parent=None):
        super().__init__(parent)
        self.scene_objects = scene_objects
        self.camera = camera
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.render_mode = "wireframe"
        self.z_buffer = []
        self.frame_buffer = None
        # Initial call to setup buffers if size is already known (e.g., fixed size widget)
        # However, resizeEvent is more reliable for dynamically sized widgets.
        self._initialize_buffers(self.width(), self.height())

    def _initialize_buffers(self, width, height):
        if width <= 0 or height <= 0:
            self.frame_buffer = None
            self.z_buffer = []
            return
        if self.frame_buffer and self.frame_buffer.width() == width and self.frame_buffer.height() == height:
            # Buffers are already correctly sized, just clear z_buffer
            self.frame_buffer.fill(Qt.GlobalColor.black)  # Clear framebuffer too
            for x_col in self.z_buffer:
                for i_y in range(len(x_col)): x_col[i_y] = float('inf')
            return

        self.frame_buffer = QImage(width, height, QImage.Format.Format_RGB32)
        self.frame_buffer.fill(Qt.GlobalColor.black)
        self.z_buffer = [[float('inf')] * height for _ in range(width)]

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = self.width()
        h = self.height()
        if w > 0 and h > 0:
            if self.camera:  # Camera might not be set during very early init
                self.camera.update_projection_matrix(w / h)
            self._initialize_buffers(w, h)
        else:
            self.frame_buffer = None
            self.z_buffer = []
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)

        if self.width() <= 0 or self.height() <= 0 or not self.camera:
            return

        # Ensure buffers are ready, especially for z-buffer mode
        if self.render_mode == "zbuffer":
            if not self.frame_buffer or self.frame_buffer.width() != self.width() or self.frame_buffer.height() != self.height():
                self._initialize_buffers(self.width(), self.height())  # Attempt re-init
            if not self.frame_buffer:  # Still no buffer, cannot paint z-buffer
                painter.setPen(Qt.GlobalColor.red)
                painter.drawText(self.rect().center() - QPointF(50, 0), "Buffer Error")
                return
            else:  # Buffers exist, ensure z-buffer is cleared for this frame
                for x_col in self.z_buffer:  # Clear z_buffer
                    for i_y in range(len(x_col)): x_col[i_y] = float('inf')
                self.frame_buffer.fill(Qt.GlobalColor.black)  # Clear framebuffer

        if self.render_mode == "wireframe":
            self._paint_wireframe(painter)
        elif self.render_mode == "zbuffer":
            self._paint_zbuffer(painter)

    def _paint_wireframe(self, painter):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for scene_obj in self.scene_objects:
            if not scene_obj.mesh or not scene_obj.mesh.vertices: continue
            scene_obj.update_transform()
            mvp_matrix = self.camera.projection_matrix * self.camera.view_matrix * scene_obj.transform
            painter.setPen(QPen(scene_obj.mesh.color, 1))

            for face_indices in scene_obj.mesh.faces:
                num_verts_in_face = len(face_indices)
                if num_verts_in_face < 2: continue
                for i in range(num_verts_in_face):
                    v_start_idx = face_indices[i]
                    v_end_idx = face_indices[(i + 1) % num_verts_in_face]
                    if not (0 <= v_start_idx < len(scene_obj.mesh.vertices) and \
                            0 <= v_end_idx < len(scene_obj.mesh.vertices)): continue
                    v_start_world = scene_obj.mesh.vertices[v_start_idx]
                    v_end_world = scene_obj.mesh.vertices[v_end_idx]
                    v_start_clip = mvp_matrix * v_start_world.to_vec4()
                    v_end_clip = mvp_matrix * v_end_world.to_vec4()
                    if v_start_clip.w <= 1e-6 or v_end_clip.w <= 1e-6: continue  # Basic W clip
                    v_start_ndc = v_start_clip.perspective_divide()
                    v_end_ndc = v_end_clip.perspective_divide()
                    x1 = (v_start_ndc.x + 1) * 0.5 * self.width()
                    y1 = (1 - (v_start_ndc.y + 1) * 0.5) * self.height()
                    x2 = (v_end_ndc.x + 1) * 0.5 * self.width()
                    y2 = (1 - (v_end_ndc.y + 1) * 0.5) * self.height()
                    painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    def _interpolate_value(self, v0_attr, v1_attr, v2_attr, w0, w1, w2):
        return v0_attr * w0 + v1_attr * w1 + v2_attr * w2

    def _paint_zbuffer(self, painter):
        if not self.frame_buffer or not self.z_buffer: return  # Already handled by paintEvent, but defensive

        for scene_obj in self.scene_objects:
            if not scene_obj.mesh or not scene_obj.mesh.vertices: continue
            scene_obj.update_transform()
            mvp_matrix = self.camera.projection_matrix * self.camera.view_matrix * scene_obj.transform

            for face_indices in scene_obj.mesh.faces:
                if len(face_indices) < 3: continue
                # Fan triangulation from vertex 0
                for i in range(len(face_indices) - 2):
                    v_idx_triangle = [face_indices[0], face_indices[i + 1], face_indices[i + 2]]
                    world_coords, clip_coords, ndc_coords, screen_coords = [], [], [], []
                    valid_triangle = True
                    for idx in v_idx_triangle:
                        if not (0 <= idx < len(scene_obj.mesh.vertices)): valid_triangle = False; break
                        world_coords.append(scene_obj.mesh.vertices[idx])
                    if not valid_triangle: continue

                    clip_coords = [mvp_matrix * v.to_vec4() for v in world_coords]
                    if any(v_clip.w <= 1e-6 for v_clip in clip_coords): continue
                    ndc_coords = [v_clip.perspective_divide() for v_clip in clip_coords]

                    for v_ndc in ndc_coords:
                        sx = (v_ndc.x + 1) * 0.5 * self.width()
                        sy = (1 - (v_ndc.y + 1) * 0.5) * self.height()
                        screen_coords.append(Vec3(sx, sy, v_ndc.z))  # Z is NDC z

                    v0_s, v1_s, v2_s = screen_coords[0], screen_coords[1], screen_coords[2]
                    min_x = math.floor(max(0, min(v0_s.x, v1_s.x, v2_s.x)))
                    max_x = math.ceil(min(self.width() - 1, max(v0_s.x, v1_s.x, v2_s.x)))
                    min_y = math.floor(max(0, min(v0_s.y, v1_s.y, v2_s.y)))
                    max_y = math.ceil(min(self.height() - 1, max(v0_s.y, v1_s.y, v2_s.y)))
                    if min_x > max_x or min_y > max_y: continue

                    # Twice signed area of triangle (P0,P1,P2)
                    area2_total = (v1_s.x - v0_s.x) * (v2_s.y - v0_s.y) - \
                                  (v1_s.y - v0_s.y) * (v2_s.x - v0_s.x)
                    if abs(area2_total) < 1e-6: continue  # Degenerate in screen space

                    # Back-face culling (optional, depends on winding consistency)
                    # if area2_total < 0: continue # If CCW is front-facing

                    for px in range(int(min_x), int(max_x) + 1):
                        for py in range(int(min_y), int(max_y) + 1):
                            # Barycentric coordinates for P=(px,py)
                            # w0 for P0, w1 for P1, w2 for P2
                            # Area(P,P1,P2)*2 / area2_total for w0
                            # Area(P0,P,P2)*2 / area2_total for w1
                            # Area(P0,P1,P)*2 / area2_total for w2

                            area2_P_P1_P2 = (v1_s.x - px) * (v2_s.y - py) - (v1_s.y - py) * (v2_s.x - px)
                            area2_P0_P_P2 = (v0_s.x - px) * (v2_s.y - py) - (v0_s.y - py) * (
                                        v2_s.x - px)  # Error in formula, use vectors

                            # P = (px, py)
                            # V0 = (v0_s.x, v0_s.y), V1 = (v1_s.x, v1_s.y), V2 = (v2_s.x, v2_s.y)
                            # w1*2*Area = (P.x - V0.x)*(V2.y - V0.y) - (P.y - V0.y)*(V2.x - V0.x)
                            # w2*2*Area = (V1.x - V0.x)*(P.y - V0.y) - (V1.y - V0.y)*(P.x - V0.x)

                            w1_num = (px - v0_s.x) * (v2_s.y - v0_s.y) - (py - v0_s.y) * (v2_s.x - v0_s.x)
                            w2_num = (v1_s.x - v0_s.x) * (py - v0_s.y) - (v1_s.y - v0_s.y) * (px - v0_s.x)

                            w1 = w1_num / area2_total
                            w2 = w2_num / area2_total
                            w0 = 1.0 - w1 - w2

                            bary_epsilon = -1e-4  # Tolerance for edge pixels
                            if w0 >= bary_epsilon and w1 >= bary_epsilon and w2 >= bary_epsilon:
                                current_z_ndc = self._interpolate_value(v0_s.z, v1_s.z, v2_s.z, w0, w1, w2)
                                if current_z_ndc < self.z_buffer[px][py]:
                                    self.z_buffer[px][py] = current_z_ndc
                                    self.frame_buffer.setPixelColor(px, py, scene_obj.mesh.color)
        painter.drawImage(0, 0, self.frame_buffer)

    def set_render_mode(self, mode_str):
        self.render_mode = mode_str
        self.update()


# ==============================================================================
# Главное Окно Приложения
# ==============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторная работа №2 - 3D Объекты")
        self.setGeometry(100, 100, 1200, 700)

        self.camera = Camera()
        self.scene_objects = []
        self.active_object_idx = 0

        self._init_ui()  # Creates render_widget, among other things
        self._create_initial_objects()  # Populates scene_objects and tabs

        # Connect render_widget properties after objects are created
        self.render_widget.camera = self.camera
        self.render_widget.scene_objects = self.scene_objects

        # Initial update for projection matrix - ensure render_widget has a size
        # QTimer.singleShot(0, self._initial_render_setup) # Defer if needed
        if self.render_widget.width() > 0 and self.render_widget.height() > 0:
            self.camera.update_projection_matrix(self.render_widget.width() / self.render_widget.height())

    # def _initial_render_setup(self): # If using QTimer.singleShot
    #     if self.render_widget.width() > 0 and self.render_widget.height() > 0:
    #         self.camera.update_projection_matrix(self.render_widget.width() / self.render_widget.height())
    #     self.render_widget.update()

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        control_panel = QFrame()
        control_panel.setFrameShape(QFrame.Shape.StyledPanel)
        control_panel.setFixedWidth(350)
        control_layout = QVBoxLayout(control_panel)

        self.btn_wireframe = QPushButton("Режим: Каркасный")
        self.btn_wireframe.clicked.connect(lambda: self.render_widget.set_render_mode("wireframe"))
        self.btn_zbuffer = QPushButton("Режим: Z-буфер (Видимые части)")
        self.btn_zbuffer.clicked.connect(lambda: self.render_widget.set_render_mode("zbuffer"))
        control_layout.addWidget(self.btn_wireframe)
        control_layout.addWidget(self.btn_zbuffer)
        control_layout.addSpacing(10)

        self.tabs = QTabWidget()
        control_layout.addWidget(self.tabs)

        # Initialize render_widget with dummy camera and objects for now
        # Will be properly set after _create_initial_objects
        self.render_widget = RenderWidget([], None, self)  # Pass self as parent
        main_layout.addWidget(self.render_widget)
        main_layout.addWidget(control_panel)

    def _create_object_controls(self, letter_name, obj_idx):
        tab = QWidget()
        layout = QGridLayout(tab)
        layout.addWidget(QLabel(f"<b>{letter_name} - Параметры:</b>"), 0, 0, 1, 2)
        h_spin = QDoubleSpinBox();
        h_spin.setRange(0.1, 10);
        h_spin.setValue(1.0);
        h_spin.setSuffix(" H")
        w_spin = QDoubleSpinBox();
        w_spin.setRange(0.01, 10);
        w_spin.setValue(1.0);
        w_spin.setSuffix(" W/SW")
        d_spin = QDoubleSpinBox();
        d_spin.setRange(0.1, 10);
        d_spin.setValue(0.3);
        d_spin.setSuffix(" D")

        if letter_name == "C":
            w_spin.setValue(0.2)
        elif letter_name == "Д":
            h_spin.setValue(1.2); w_spin.setValue(1.0)

        btn_apply_dims = QPushButton("Применить размеры")
        btn_apply_dims.clicked.connect(lambda chk, oi=obj_idx, hs=h_spin, ws=w_spin, ds=d_spin: \
                                           self._update_object_dimensions(oi, hs.value(), ws.value(), ds.value()))
        layout.addWidget(QLabel("Высота:"), 1, 0);
        layout.addWidget(h_spin, 1, 1)
        layout.addWidget(QLabel("Ширина(Д)/Толщ.(C):"), 2, 0);
        layout.addWidget(w_spin, 2, 1)
        layout.addWidget(QLabel("Глубина:"), 3, 0);
        layout.addWidget(d_spin, 3, 1)
        layout.addWidget(btn_apply_dims, 4, 0, 1, 2)

        if obj_idx == 0:
            self.obj_c_dims_spins = (h_spin, w_spin, d_spin)
        elif obj_idx == 1:
            self.obj_d_dims_spins = (h_spin, w_spin, d_spin)

        layout.addWidget(QLabel(f"<b>{letter_name} - Трансформации:</b>"), 5, 0, 1, 2)
        trans_step, rot_step = 0.1, math.radians(10)
        buttons_data = [
            ("Перемещение:", [("X+", Vec3(trans_step, 0, 0)), ("X-", Vec3(-trans_step, 0, 0)),
                              ("Y+", Vec3(0, trans_step, 0)), ("Y-", Vec3(0, -trans_step, 0)),
                              ("Z+", Vec3(0, 0, trans_step)), ("Z-", Vec3(0, 0, -trans_step))]),
            ("Вращение:", [("RotX+", Vec3(rot_step, 0, 0)), ("RotX-", Vec3(-rot_step, 0, 0)),
                           ("RotY+", Vec3(0, rot_step, 0)), ("RotY-", Vec3(0, -rot_step, 0)),
                           ("RotZ+", Vec3(0, 0, rot_step)), ("RotZ-", Vec3(0, 0, -rot_step))])
        ]
        row = 6
        for label_text, actions in buttons_data:
            layout.addWidget(QLabel(label_text), row, 0, 1, 2);
            row += 1
            for i in range(0, len(actions), 2):
                btn1_text, val1 = actions[i]
                btn1 = QPushButton(btn1_text)
                mode = "translate" if "Rot" not in btn1_text else "rotate"
                btn1.clicked.connect(lambda chk, oi=obj_idx, m=mode, v=val1: self._transform_object(oi, m, v))
                layout.addWidget(btn1, row, 0)
                if i + 1 < len(actions):
                    btn2_text, val2 = actions[i + 1]
                    btn2 = QPushButton(btn2_text)
                    mode2 = "translate" if "Rot" not in btn2_text else "rotate"
                    btn2.clicked.connect(lambda chk, oi=obj_idx, m=mode2, v=val2: self._transform_object(oi, m, v))
                    layout.addWidget(btn2, row, 1)
                row += 1
        layout.setRowStretch(row, 1)
        return tab

    def _create_camera_controls(self):
        cam_tab = QWidget()
        layout = QGridLayout(cam_tab)
        layout.addWidget(QLabel("<b>Управление камерой:</b>"), 0, 0, 1, 2)
        move_s, rot_s, zoom_in, zoom_out = 0.2, math.radians(5), 0.9, 1.1

        def cam_act(func):
            func(); self.render_widget.update()

        cam_buttons = [
            ("Перемещение (Оси мира):", [
                ("Влево (A)", lambda: self.camera.move(-move_s, 0, 0)),
                ("Вправо (D)", lambda: self.camera.move(move_s, 0, 0)),
                ("Вверх (W)", lambda: self.camera.move(0, move_s, 0)),
                ("Вниз (S)", lambda: self.camera.move(0, -move_s, 0))]),
            ("Зум (к цели):", [
                ("Приблизить (Z)", lambda: self.camera.zoom(zoom_in)),
                ("Отдалить (X)", lambda: self.camera.zoom(zoom_out))]),
            ("Вращение орбиты (вокруг цели):", [
                ("Рыскание + (Q)", lambda: self.camera.rotate_orbit(0, rot_s)),
                ("Рыскание - (E)", lambda: self.camera.rotate_orbit(0, -rot_s))])
        ]
        row = 1
        for label_text, actions in cam_buttons:
            layout.addWidget(QLabel(label_text), row, 0, 1, 2);
            row += 1
            for i in range(0, len(actions), 2):
                btn1_text, func1 = actions[i];
                btn1 = QPushButton(btn1_text);
                btn1.clicked.connect(lambda chk, f=func1: cam_act(f));
                layout.addWidget(btn1, row, 0)
                if i + 1 < len(actions):
                    btn2_text, func2 = actions[i + 1];
                    btn2 = QPushButton(btn2_text);
                    btn2.clicked.connect(lambda chk, f=func2: cam_act(f));
                    layout.addWidget(btn2, row, 1)
                row += 1
        btn_reset = QPushButton("Сброс камеры");
        btn_reset.clicked.connect(lambda: cam_act(self._reset_camera_view));
        layout.addWidget(btn_reset, row, 0, 1, 2);
        row += 1
        layout.setRowStretch(row, 1)
        return cam_tab

    def _create_initial_objects(self):
        self.scene_objects.clear()
        tab_c_controls = self._create_object_controls("C", 0)
        tab_d_controls = self._create_object_controls("Д", 1)

        if hasattr(self, 'obj_c_dims_spins'):
            h, w, d = self.obj_c_dims_spins[0].value(), self.obj_c_dims_spins[1].value(), self.obj_c_dims_spins[
                2].value()
            obj_c = SceneObject(create_letter_c_mesh(h, w, d));
            obj_c.translation = Vec3(-0.8, 0, 0)
            self.scene_objects.append(obj_c);
            self.tabs.addTab(tab_c_controls, "Буква C")
        if hasattr(self, 'obj_d_dims_spins'):
            h, w, d = self.obj_d_dims_spins[0].value(), self.obj_d_dims_spins[1].value(), self.obj_d_dims_spins[
                2].value()
            obj_d = SceneObject(create_letter_d_mesh(w, h, d));
            obj_d.translation = Vec3(0.8, 0, 0)
            self.scene_objects.append(obj_d);
            self.tabs.addTab(tab_d_controls, "Буква Д")

        self.tabs.addTab(self._create_camera_controls(), "Камера")
        self.active_object_idx = 0 if self.scene_objects else -1
        self.tabs.setCurrentIndex(0)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.render_widget.update()

    def _on_tab_changed(self, index):
        if 0 <= index < len(self.scene_objects): self.active_object_idx = index

    def _update_object_dimensions(self, obj_idx, h_val, w_val, d_val):
        if not (0 <= obj_idx < len(self.scene_objects)): return
        obj = self.scene_objects[obj_idx]
        # Determine if it's C or D based on index (fragile) or stored type/name
        if obj_idx == 0:  # Assuming C is always 0
            obj.mesh = create_letter_c_mesh(h_val, w_val, d_val)  # w_val is stroke_width
        elif obj_idx == 1:  # Assuming D is always 1
            obj.mesh = create_letter_d_mesh(w_val, h_val, d_val)  # w_val is width
        self.render_widget.update()

    def _transform_object(self, obj_idx, mode, value: Vec3):
        if not (0 <= obj_idx < len(self.scene_objects)): return
        obj = self.scene_objects[obj_idx]
        if mode == "translate":
            obj.translation = obj.translation + value
        elif mode == "rotate":
            obj.rotation = obj.rotation + value
        self.render_widget.update()  # update_transform is called in paint

    def _reset_camera_view(self):
        self.camera.eye = Vec3(0, 0, 5);
        self.camera.target = Vec3(0, 0, 0);
        self.camera.up = Vec3(0, 1, 0)
        self.camera.update_view_matrix()

    def keyPressEvent(self, event):
        handled = False
        cam_s, cam_r, cam_zi, cam_zo = 0.1, math.radians(3), 0.95, 1.05
        key_actions = {
            Qt.Key.Key_A: lambda: self.camera.move(-cam_s, 0, 0), Qt.Key.Key_D: lambda: self.camera.move(cam_s, 0, 0),
            Qt.Key.Key_W: lambda: self.camera.move(0, cam_s, 0), Qt.Key.Key_S: lambda: self.camera.move(0, -cam_s, 0),
            Qt.Key.Key_Z: lambda: self.camera.zoom(cam_zi), Qt.Key.Key_X: lambda: self.camera.zoom(cam_zo),
            Qt.Key.Key_Q: lambda: self.camera.rotate_orbit(0, cam_r),
            Qt.Key.Key_E: lambda: self.camera.rotate_orbit(0, -cam_r),
        }
        action = key_actions.get(event.key())
        if action: action(); handled = True; self.render_widget.update()
        if not handled: super().keyPressEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())