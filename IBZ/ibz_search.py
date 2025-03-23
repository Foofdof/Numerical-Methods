import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

# ========================
# 1. Задание параметров
# ========================
w = 20.0
h = w/10
l = w/2
alpha_deg = 60.0  # угол в градусах
alpha = np.deg2rad(alpha_deg)

# ========================
# 2. Определение исходного многоугольника
# ========================
# Таблица координат узлов исходного полигона (в порядке обхода)
coords = [
    (0, l * np.sin(alpha) - h/2),
    (-w/2 - h + h/2 * (np.cos(alpha)/np.sin(alpha)), l * np.sin(alpha) - h/2),
    (-w/2 + l * np.cos(alpha) - h - h/2 * (np.cos(alpha)/np.sin(alpha)), h/2),
    (-w + l * np.cos(alpha) + h * np.tan(alpha) - 2*h - h/2 * (np.cos(alpha)/np.sin(alpha)), h/2),
    (-w + l * np.cos(alpha) + h * np.tan(alpha) - 2*h - h/2 * (np.cos(alpha)/np.sin(alpha)), 0),
    (-w/2 + l * np.cos(alpha) + h/np.sin(alpha) - h, 0),
    (-w/2 + h * np.tan(alpha) - h, l * np.sin(alpha) - h),
    (0, l * np.sin(alpha) - h),
    (0, l * np.sin(alpha))
]

poly_orig = Polygon(coords)

# ========================
# 3. Получение полной ячейки путём отражений
# ========================
def reflect_vertical(poly):
    """Отражение относительно вертикальной оси: (x,y) -> (-x,y)"""
    return Polygon([(-x, y) for (x, y) in poly.exterior.coords])

def reflect_horizontal(poly):
    """Отражение относительно горизонтальной оси: (x,y) -> (x,-y)"""
    return Polygon([(x, -y) for (x, y) in poly.exterior.coords])

poly_vert = reflect_vertical(poly_orig)
poly_horiz = reflect_horizontal(poly_orig)
poly_both = reflect_horizontal(poly_vert)  # отражение и по вертикали, и по горизонтали

# Используем buffer(0) и unary_union для корректного объединения
poly_list = [
    poly_orig.buffer(0),
    poly_vert.buffer(0),
    poly_horiz.buffer(0),
    poly_both.buffer(0)
]
full_cell = unary_union(poly_list)

# ========================
# 4. Вычисление базисных векторов и обратной решётки
# ========================
minx, miny, maxx, maxy = full_cell.bounds
Lx = maxx - minx
Ly = maxy - miny
print("Полная ячейка, bounds:", full_cell.bounds)
print("Lx =", Lx, "Ly =", Ly)

# Предположим, что базисные векторы описывают прямоугольную ячейку:
a1 = np.array([Lx, 0])
a2 = np.array([0, Ly])

# Для прямоугольной ячейки обратные базисные векторы:
b1 = np.array([2 * np.pi / Lx, 0])
b2 = np.array([0, 2 * np.pi / Ly])
print("Обратные базисные векторы:")
print("b1 =", b1)
print("b2 =", b2)

# ========================
# 5. Построение первой зоны Брюилена (Wigner–Zeyt cell)
# ========================
N = 2
reciprocal_points = []
for m in range(-N, N + 1):
    for n in range(-N, N + 1):
        pt = m * b1 + n * b2
        reciprocal_points.append(pt)
reciprocal_points = np.array(reciprocal_points)

vor = Voronoi(reciprocal_points)

origin_index = None
for i, pt in enumerate(reciprocal_points):
    if np.allclose(pt, [0, 0], atol=1e-6):
        origin_index = i
        break
if origin_index is None:
    raise ValueError("Не найдена точка (0,0) в обратной решётке.")

region_index = vor.point_region[origin_index]
region = vor.regions[region_index]
if -1 in region:
    raise ValueError("Область не ограничена. Попробуйте увеличить число периодов N.")

bz_vertices = vor.vertices[region]
bz_polygon = Polygon(bz_vertices)

# ========================
# 6. Выделение irreducible Brillouin zone (IBZ)
# ========================
halfplane_kx = box(0, -1e6, 1e6, 1e6)  # область: kx >= 0
halfplane_ky = box(-1e6, 0, 1e6, 1e6)  # область: ky >= 0
IBZ = bz_polygon.intersection(halfplane_kx).intersection(halfplane_ky)

# Вывод границ IBZ (координаты вершин)
if not IBZ.is_empty:
    if IBZ.geom_type == 'Polygon':
        ibz_coords = list(IBZ.exterior.coords)
        print("Границы IBZ:")
        for pt in ibz_coords:
            print(pt)
    elif IBZ.geom_type == 'MultiPolygon':
        for i, poly in enumerate(IBZ.geoms):
            ibz_coords = list(poly.exterior.coords)
            print(f"Границы IBZ для полигона {i}:")
            for pt in ibz_coords:
                print(pt)
else:
    print("IBZ пустая!")

# ========================
# 7. Визуализация
# ========================
fig, axs = plt.subplots(1, 2, figsize=(16, 8))

# 7.1. Отрисовка структуры (полной ячейки) с внутренними и внешними границами и базисными векторами
axs[0].set_title("Полная ячейка и базисные векторы")
def plot_polygon_boundaries(ax, poly, ext_label=None, int_label=None):
    x_ext, y_ext = poly.exterior.xy
    ax.plot(x_ext, y_ext, 'k-', lw=2, label=ext_label)
    for i, interior in enumerate(poly.interiors):
        x_int, y_int = interior.xy
        label = int_label if i == 0 else None
        ax.plot(x_int, y_int, 'k--', lw=2, label=label)

if full_cell.geom_type == 'Polygon':
    plot_polygon_boundaries(axs[0], full_cell, ext_label="Внешняя граница", int_label="Внутренняя граница")
elif full_cell.geom_type == 'MultiPolygon':
    for i, geom in enumerate(full_cell.geoms):
        lab_ext = "Внешняя граница" if i == 0 else None
        lab_int = "Внутренняя граница" if i == 0 else None
        plot_polygon_boundaries(axs[0], geom, ext_label=lab_ext, int_label=lab_int)

x_orig, y_orig = poly_orig.exterior.xy
axs[0].plot(x_orig, y_orig, 'b--', lw=2, label="Исходный полигон")

# Отрисовка базисных векторов для реальной ячейки (от центра ограничивающего прямоугольника)
center = np.array([(minx + maxx)/2, (miny + maxy)/2])
axs[0].arrow(center[0], center[1], a1[0], a1[1],
             head_width=0.5, head_length=1, fc='magenta', ec='magenta', lw=2, label='$a_1$')
axs[0].arrow(center[0], center[1], a2[0], a2[1],
             head_width=0.5, head_length=1, fc='orange', ec='orange', lw=2, label='$a_2$')

axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].axis('equal')
axs[0].legend()

# 7.2. Отрисовка обратного пространства, IBZ и базисных векторов обратной решётки
axs[1].set_title("Первая зона Брюилена, IBZ и базисные векторы")
x_bz, y_bz = bz_polygon.exterior.xy
axs[1].plot(x_bz, y_bz, 'b-', lw=2, label='Первая зона Брюилена')
if not IBZ.is_empty:
    if IBZ.geom_type == 'Polygon':
        x_ibz, y_ibz = IBZ.exterior.xy
        axs[1].fill(x_ibz, y_ibz, color='red', alpha=0.5, label='Irreducible BZ')
    else:
        for geom in IBZ.geoms:
            x_ibz, y_ibz = geom.exterior.xy
            axs[1].fill(x_ibz, y_ibz, color='red', alpha=0.5, label='Irreducible BZ')
axs[1].plot(reciprocal_points[:, 0], reciprocal_points[:, 1], 'ko', markersize=3, label='Обратная решётка')

origin = np.array([0, 0])
axs[1].arrow(origin[0], origin[1], b1[0], b1[1],
             head_width=0.03, head_length=0.05, fc='magenta', ec='magenta', lw=2, label='$b_1$')
axs[1].arrow(origin[0], origin[1], b2[0], b2[1],
             head_width=0.03, head_length=0.05, fc='orange', ec='orange', lw=2, label='$b_2$')

axs[1].axhline(0, color='gray', linestyle='--')
axs[1].axvline(0, color='gray', linestyle='--')
axs[1].set_xlabel('$k_x$')
axs[1].set_ylabel('$k_y$')
axs[1].legend()
axs[1].axis('equal')

plt.suptitle("Построение структуры, базисных векторов и irreducible Brillouin zone", fontsize=16)
plt.show()
