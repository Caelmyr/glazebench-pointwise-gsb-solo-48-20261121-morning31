from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from .geometry import DEFAULT_CORNERS, canvas_dimensions, homography
from .photos import linear_to_srgb, png_bytes, srgb_to_linear
from .store import make_id
from .photos import store_capture


DEMO_KEY = "glazebench.training.v1"
PLANE_SIZE = (620, 472)
PHOTO_SIZE = (766, 598)
DEMO_ROLES = ("diffuse", "parallel", "cross", "rake-left", "rake-right")
POSES = {
    "diffuse": [[0.092, 0.086], [0.911, 0.112], [0.894, 0.912], [0.081, 0.889]],
    "parallel": [[0.122, 0.107], [0.910, 0.084], [0.925, 0.871], [0.101, 0.916]],
    "cross": [[0.079, 0.101], [0.908, 0.119], [0.877, 0.916], [0.081, 0.909]],
    "rake-left": [[0.089, 0.094], [0.894, 0.088], [0.902, 0.905], [0.064, 0.911]],
    "rake-right": [[0.094, 0.095], [0.913, 0.094], [0.926, 0.906], [0.112, 0.893]],
}


@dataclass(frozen=True)
class DemoSpecimen:
    seed: int
    name: str
    body: str
    glaze: str
    firing: str
    batch: str
    colors: tuple
    bare: tuple
    fine: float
    coat_line: float
    burn: float


SERIES = (
    DemoSpecimen(81, "Tidepool 07", "Light stoneware", "Tidepool satin", "Reduction · R04", "Fired studies A",
                ((33, 91, 80), (97, 150, 134), (170, 181, 139)), (157, 112, 82), 0.034, 0.83, 0.14),
    DemoSpecimen(114, "Honey ash 03", "Speckled stoneware", "Honey ash", "Oxidation · O11", "Fired studies B",
                ((120, 56, 36), (186, 124, 56), (229, 177, 97)), (142, 98, 74), 0.041, 0.75, 0.30),
    DemoSpecimen(129, "Plum wash 12", "Light stoneware", "Plum wash", "Oxidation · O11", "Fired studies B",
                ((75, 56, 75), (138, 104, 135), (185, 155, 158)), (157, 111, 83), 0.026, 0.81, 0.23),
    DemoSpecimen(94, "Tidepool 11", "Light stoneware", "Tidepool satin", "Oxidation · O11", "Fired studies A",
                ((42, 101, 94), (113, 158, 145), (188, 197, 165)), (154, 116, 89), 0.029, 0.78, 0.08),
)


def random_field(generator, size, y_space, x_space):
    height, width = size
    rows, cols = int(np.ceil(height / y_space)) + 4, int(np.ceil(width / x_space)) + 4
    coarse = generator.normal(size=(rows, cols)).astype(np.float32)
    y, x = np.indices(size, dtype=np.float32)
    field = ndimage.map_coordinates(coarse, (y / y_space + 1, x / x_space + 1), order=3)
    return field.astype(np.float32)


@dataclass(frozen=True)
class CeramicField:
    diffuse: np.ndarray
    height: np.ndarray
    glaze: np.ndarray
    graphite_marks: np.ndarray
    roughness: np.ndarray
    u: np.ndarray
    v: np.ndarray


def surface_palette(colors, values):
    linear_colors = srgb_to_linear(np.array(colors, dtype=np.float32) / 255)
    stops = (-1.2, 0.05, 1.25)
    return np.stack(
        [np.interp(values, stops, linear_colors[:, channel]) for channel in range(3)],
        axis=2,
    ).astype(np.float32)


def glaze_boundary(field, x, y, boundary):
    edge = boundary + 0.024 * np.sin(9 * x + 0.2) + 0.014 * field
    transition = np.clip((edge - y) / 0.035 + 0.48, 0, 1)
    return transition**2 * (3 - 2 * transition)


def bare_clay(generator, shape, color):
    low = random_field(generator, shape, 40, 51)
    clay = srgb_to_linear(np.asarray(color, dtype=np.float32) / 255)
    speckle = generator.random(shape) > 0.997
    speckle = ndimage.gaussian_filter(speckle.astype(np.float32), 0.65)
    modulation = 0.99 + low * 0.06 - speckle * 1.3
    return np.maximum(clay[None, None, :] * modulation[:, :, None], 0)


def bubble_field(generator, shape, v, u):
    values = np.zeros(shape, dtype=np.float32)
    glaze_holes = np.zeros(shape, dtype=np.float32)
    for index in range(61):
        cy = generator.uniform(0.06, 0.73)
        cx = generator.uniform(0.06, 0.93)
        radius_px = generator.uniform(1.4, 5.3) if index < 18 else generator.uniform(0.6, 1.8)
        ys = (v - cy) * shape[0]
        xs = (u - cx) * shape[1]
        radius2 = (xs**2 + ys**2) / radius_px**2
        cup = np.exp(-radius2 * 1.2)
        ring = np.exp(-((radius2**0.5 - 1.1) * 3.1) ** 2)
        values += (-cup + ring * 0.24) * generator.uniform(0.23, 0.70)
        glaze_holes = np.maximum(glaze_holes, cup * generator.uniform(0.14, 0.88))
    return values, glaze_holes


def raised_strokes(x, y, fine):
    sweep = x * 6.5 + y * 54 + 0.38 * np.sin(6 * x + 3 * y) + 0.23 * np.sin(12 * x)
    ridges = np.sin(sweep) * 0.10 + np.sin(sweep * 2.3 + 0.2) * 0.022
    thin = 0.95 * np.exp(-((y - 0.315 - 0.05 * np.sin(9 * x)) / 0.004) ** 2)
    thin *= np.clip(1 - np.abs(x - 0.60) * 4.1, 0, 1)
    return ridges * fine * 30, thin


def neutral_patch(diffuse, roughness, x, y):
    sample = (x >= 0.07) & (x <= 0.165) & (y >= 0.844) & (y <= 0.961)
    border = (x >= 0.063) & (x <= 0.172) & (y >= 0.835) & (y <= 0.968) & ~sample
    rgb = np.array([0.194, 0.172, 0.155], dtype=np.float32)
    diffuse[sample] = rgb[None, :] * (0.97 + 0.042 * (x[sample] - 0.10))[:, None]
    diffuse[border] *= 0.70
    roughness[sample] = 0.76
    return sample


def crackle_draw(generator, x, y, shape, seed):
    if seed not in (81, 129):
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
    coarse = generator.uniform(-0.07, 1.07, size=(360 if seed == 129 else 235, 2))
    jitter_x = 0.005 * np.sin(y * 75 + x * 11) + 0.004 * np.sin(y * 24)
    jitter_y = 0.005 * np.sin(x * 59 + y * 14)
    query = np.stack(((x + jitter_x).ravel(), (y + jitter_y).ravel()), axis=1)
    distances, nearest = cKDTree(coarse).query(query, k=2)
    pair_span = np.linalg.norm(coarse[nearest[:, 1]] - coarse[nearest[:, 0]], axis=1)
    radius = 0.0014 if seed == 129 else 0.0011
    margin = (distances[:, 1] ** 2 - distances[:, 0] ** 2) / np.maximum(pair_span, 0.006) / 2
    score = np.exp(-((margin / radius) ** 2)).reshape(shape)
    exposure = np.clip(random_field(generator, shape, 73, 61) * 0.28 + 0.47, 0.17, 0.81)
    density = 1.0 if seed == 129 else 0.69
    shadow = np.clip(score * density * exposure, 0, 0.83)
    edge = np.exp(-((margin / (radius * 2.8)) ** 2)).reshape(shape) - score
    return shadow.astype(np.float32), edge.astype(np.float32)


def bake_field(specimen):
    width, height = PLANE_SIZE
    shape = (height, width)
    generator = np.random.default_rng(specimen.seed)
    y, x = np.indices(shape, dtype=np.float32)
    x /= width
    y /= height
    broad = random_field(generator, shape, 82, 114)
    middle = random_field(generator, shape, 22, 40)
    small = random_field(generator, shape, 4.2, 4.9)
    micro = generator.normal(size=shape).astype(np.float32)
    hills = 0.28 * broad + 0.13 * middle + 0.08 * small + 0.012 * micro
    ridges, scratch = raised_strokes(x, y, specimen.fine)
    cup_height, pits = bubble_field(generator, shape, y, x)
    oxides = 0.40 * broad + 0.22 * middle + 0.055 * small + 0.3 * ridges
    temperature = 0.33 * np.exp(-((y - 0.07) / 0.16) ** 2)
    surface = surface_palette(specimen.colors, oxides + temperature - specimen.burn * y)
    body = bare_clay(generator, shape, specimen.bare)
    fracture, fracture_rim = crackle_draw(generator, x, y, shape, specimen.seed)
    cover = glaze_boundary(broad, x, y, specimen.coat_line)
    thickening = 0.12 * np.exp(-((y - specimen.coat_line + 0.026) / 0.022) ** 2)
    diffuse = cover[:, :, None] * surface * (1 + thickening[:, :, None]) + (1 - cover[:, :, None]) * body
    dark = np.clip((pits * 0.78 + scratch * 0.085 + fracture) * cover, 0, 0.92)
    shiver = 1 - 0.18 * np.maximum(0, -small - 1.4) * cover
    diffuse *= ((1 - dark) * shiver * (1 + specimen.fine * micro))[:, :, None]
    roughness = np.clip(0.45 + 0.4 * (1 - cover) + small * 0.08, 0.20, 0.92)
    neutral_patch(diffuse, roughness, x, y)
    return CeramicField(
        diffuse=np.clip(diffuse, 0, 1),
        height=hills + ridges * 0.34 + (cup_height - fracture * 0.32 + fracture_rim * 0.10) * cover + scratch * 0.38,
        glaze=cover,
        graphite_marks=dark,
        roughness=roughness,
        u=x,
        v=y,
    )


def gloss_reflection(field, role):
    x, y = field.u, field.v
    hotspot = np.exp(-(((x - 0.64 + 0.17 * y) / 0.21) ** 2 + ((y - 0.38) / 0.25) ** 2))
    elongated = np.exp(-(((x - 0.64 - 0.38 * (y - 0.3)) / 0.075) ** 2 + ((y - 0.49) / 0.31) ** 2))
    low = (1 - field.roughness) * field.glaze * (0.5 + 0.5 * np.sin(8 * x + 13 * y) ** 2)
    intensity = 0.079 * low if role == "diffuse" else 0.34 * low + 0.24 * hotspot + 0.12 * elongated
    if role == "cross":
        intensity = (0.009 + hotspot * 0.018) * low
    return intensity * (1 - field.graphite_marks * 0.58)


def light_field(field, role):
    x, y = field.u, field.v
    dy, dx = np.gradient(0.92 * field.height)
    broad = 0.93 + 0.13 * (1 - y) - 0.08 * (x - 0.5) ** 2
    if role in ("rake-left", "rake-right"):
        direction = 1 if role == "rake-left" else -1
        side = 0.91 + direction * 0.13 * (0.5 - x)
        shading = np.clip(0.66 + (dx * direction - dy * 0.23) * 3.9, 0.26, 1.32)
        rgb = field.diffuse * (broad * side * shading)[:, :, None]
    else:
        side = np.clip(0.98 + dx * 0.38 - dy * 0.31, 0.64, 1.2)
        reflection = gloss_reflection(field, role)
        exposure = 0.88 if role == "cross" else 1.09 if role == "parallel" else 1.0
        rgb = field.diffuse * (broad * side * exposure)[:, :, None]
        rgb += reflection[:, :, None] * np.array([0.69, 0.79, 0.82], dtype=np.float32)
    return np.clip(rgb, 0, 1)


def desk_background(width, height, seed):
    generator = np.random.default_rng(seed + 500)
    noise = random_field(generator, (height, width), 21, 30) * 0.018
    y, x = np.indices((height, width), dtype=np.float32)
    grad = 0.12 + 0.19 * np.exp(-((x / width - 0.24) ** 2 + (y / height - 0.10) ** 2) * 2.4)
    grain = generator.normal(size=(height, width)).astype(np.float32) * 0.008
    neutral = np.maximum(grad + grain + noise, 0.035)
    return neutral[:, :, None] * np.array([0.50, 0.57, 0.58], dtype=np.float32)


def tile_alpha(x, y):
    horizontal = np.abs(x - 0.5) - 0.461
    vertical = np.abs(y - 0.5) - 0.452
    corner = np.hypot(np.maximum(horizontal, 0), np.maximum(vertical, 0))
    nearest = np.minimum(np.maximum(horizontal, vertical), 0)
    sdf = corner + nearest - 0.035
    return np.clip(0.5 - sdf * 410, 0, 1)


def warp_to_photo(array, corners, photo_size, seed):
    width, height = photo_size
    photo_to_tile = np.linalg.inv(homography(DEFAULT_CORNERS, corners))
    photo_y, photo_x = np.indices((height, width), dtype=np.float32)
    photo_y = (photo_y + 0.5) / height
    photo_x = (photo_x + 0.5) / width
    denominator = photo_to_tile[2, 0] * photo_x + photo_to_tile[2, 1] * photo_y + photo_to_tile[2, 2]
    tile_u = (photo_to_tile[0, 0] * photo_x + photo_to_tile[0, 1] * photo_y + photo_to_tile[0, 2]) / denominator
    tile_v = (photo_to_tile[1, 0] * photo_x + photo_to_tile[1, 1] * photo_y + photo_to_tile[1, 2]) / denominator
    tile_h, tile_w = array.shape[:2]
    coords = (tile_v * tile_h - 0.5, tile_u * tile_w - 0.5)
    tile_image = np.stack(
        [ndimage.map_coordinates(array[:, :, component], coords, order=1, mode="nearest") for component in range(3)],
        axis=2,
    )
    alpha = tile_alpha(tile_u, tile_v)
    shadow = ndimage.shift(ndimage.gaussian_filter(alpha, 7), (9, 5), order=1, mode="constant")
    stage = desk_background(width, height, seed)
    stage *= (1 - 0.61 * shadow)[:, :, None]
    rim = np.clip(alpha - ndimage.minimum_filter(alpha, size=8), 0, 1)
    tile_image *= 1 - 0.24 * rim[:, :, None]
    mixed = stage * (1 - alpha[:, :, None]) + tile_image * alpha[:, :, None]
    return linear_to_srgb(mixed)


def render_photo(field, role, specimen):
    linear = light_field(field, role)
    image = warp_to_photo(linear, POSES[role], PHOTO_SIZE, specimen.seed)
    array = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(array)


def rectangle(name, purpose, left, top, right, bottom):
    return {
        "id": make_id(), "name": name, "purpose": purpose,
        "vertices": [[left, top], [right, top], [right, bottom], [left, bottom]],
    }


def training_regions(specimen):
    return [
        rectangle("Kiln face", "inspection", 0.045, 0.050, 0.952, 0.809),
        rectangle("Upper body", "glazed", 0.16, 0.17, 0.38, 0.36),
        rectangle("Exposed foot", "bare", 0.43, min(specimen.coat_line + 0.085, 0.917), 0.79, 0.958),
        rectangle("Neutral card", "neutral", 0.086, 0.858, 0.15, 0.943),
        rectangle("Bottom marking", "ignore", 0.22, 0.931, 0.39, 0.981),
    ]


def training_specimen(store, specimen):
    field = bake_field(specimen)
    photos = []
    for role in DEMO_ROLES:
        image = render_photo(field, role, specimen)
        photo = store_capture(
            store.blobs, png_bytes(image), role,
            f"{specimen.name.lower().replace(' ', '-')}-{role}.png",
            "Procedurally rendered ceramic training photograph.",
            make_id(),
            synthetic=True,
        )
        photos.append(photo)
    base = photos[0]
    size = canvas_dimensions(base, POSES["diffuse"], 94, 71)
    return {
        "name": specimen.name,
        "batch": specimen.batch,
        "body": specimen.body,
        "glaze": specimen.glaze,
        "firing": specimen.firing,
        "notes": "Synthetic training tile. The footprint and the optics are procedurally generated.",
        "width_mm": 94,
        "height_mm": 71,
        "synthetic": True,
        "photos": photos,
        "plane": {"photo_id": base["id"], "corners": POSES["diffuse"], "size": list(size)},
        "regions": training_regions(specimen),
        "registrations": {
            photo["id"]: {"base_photo_id": base["id"], "corners": POSES[photo["role"]], "support_percent": 100}
            for photo in photos[1:]
        },
    }


def load_training(store):
    existing = store.seeded_tiles(DEMO_KEY)
    if existing:
        return existing, False
    rows = [training_specimen(store, specimen) for specimen in SERIES]
    return store.seed_tiles(DEMO_KEY, rows), True


@lru_cache(maxsize=1)
def preview_photo():
    specimen = SERIES[0]
    rendered = render_photo(bake_field(specimen), "diffuse", specimen)
    return png_bytes(rendered)
