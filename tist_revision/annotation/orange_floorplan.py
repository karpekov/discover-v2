"""Orange4Home floor plan in the style of the re-rendered CASAS plans (`metadata/floor_plans_augmented/`).

Walls, door openings and furniture footprints are the vector geometry of the two Amiqual4Home plans
(https://amiqual4home.inria.fr/en/orange4home/, `ground_floor.pdf` / `first_floor.pdf`, in PDF points), redrawn
with thick black walls, gray furniture blocks and the CASAS marker set, both floors stacked in one image.

Orange4Home publishes only the room of each sensor (`sensors_localisation.txt`), not its position. Positions
inside a room are therefore schematic: presence sensors over the area they are named after, contact sensors on
their door / cupboard / drawer, shutters on the windows, lights on the ceiling of their room, and wall-switch
panels where the event stream puts them (the door or sensor that fires right before and after a press).

    python -m tist_revision.annotation.orange_floorplan   # writes orange.png / orange.svg, updates the metadata
"""
import base64
import json
import math
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
METADATA = PROJECT_ROOT / 'metadata' / 'casas_metadata.json'
FLOOR_PLANS = PROJECT_ROOT / 'metadata' / 'floor_plans_augmented'
CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
FONT_CACHE = Path.home() / '.cache' / 'discover-v2' / 'fonts'
FONT_URL = 'https://github.com/google/fonts/raw/main/ofl/inter/'
FONTS = {'normal': ('InterVariable.ttf', 'Inter%5Bopsz%2Cwght%5D.ttf'),
         'italic': ('InterVariable-Italic.ttf', 'Inter-Italic%5Bopsz%2Cwght%5D.ttf')}
FONT = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"

S = 3.0            # pixels per PDF point
M = 70             # margin
W = round(2 * M + 842 * S)
OY = {'first': 150, 'ground': 1610}
H = OY['ground'] + round(387 * S) + 150
WALL = 11
INK, FURN, FURN_LIGHT, PILLOW = '#000000', '#d1d5db', '#f3f4f6', '#9ca3af'
ROOM, TEXT, TITLE, DOOR, DASH, SHUTTER = '#9ca3af', '#6b7280', '#374151', '#c2410c', '#94a3b8', '#64748b'

# ------------------------------------------------------------------ geometry (PDF points, per floor)

WALLS = {
    'ground': [
        [(4.05, 5.35), (4.05, 386)],
        [(4.05, 5.35), (509.5, 5.35), (560.5, 34.85), (622.1, 96.4), (777, 364.65), (88.75, 364.65)],
        [(88.75, 5.35), (88.75, 172.3)],
        [(168.4, 364.65), (168.4, 260.1), (88.75, 260.1), (88.75, 386)],
        [(229.1, 306.4), (431.9, 306.4), (431.9, 364.65)],
    ],
    'first': [
        [(3.65, 19.75), (536.2, 19.75), (645, 84), (681.7, 147.7), (835, 413.25), (3.65, 413.25), (3.65, 19.75)],
        [(489.85, 19.75), (489.85, 264.45)],
        [(489.85, 170.7), (695, 170.7)],
        [(3.65, 264.45), (489.85, 264.45)],
        [(176.35, 264.45), (176.35, 413.25)],
        [(464.25, 264.45), (464.25, 413.25)],
    ],
}

# doors with a contact sensor: sensor -> (floor, end, end)
SENSOR_DOORS = {
    'entrance_door': ('ground', (10.7, 383), (83.8, 383)),
    'bedroom_closet_door': ('first', (489.85, 24.5), (489.85, 91.3)),
    'bedroom_door': ('first', (386.8, 264.45), (460, 264.45)),
    'bathroom_door': ('first', (176.35, 269.5), (176.35, 339.1)),
    'office_door': ('first', (464.25, 269.5), (464.25, 342.4)),
}
# doors without a sensor: (floor, opening end, opening end, hinge, closed tip, open tip)
PLAIN_DOORS = [
    ('ground', (4.05, 9.5), (4.05, 79.6), (7.1, 10.5), (7.1, 75.6), (72.2, 10.5)),
    ('ground', (88.75, 264.5), (88.75, 312.6), (91.8, 310), (91.8, 266.2), (135.5, 310)),
    ('ground', (302.9, 5.35), (376, 5.35), (306.9, 8.4), (372, 8.4), (306.9, 73.5)),
]
# windows and glazed doors: (floor, end, end, shutter sensor or None); the shutter is drawn on the room side
WINDOWS = [
    ('ground', (419.7, 5.35), (492.4, 5.35), 'livingroom_shutter1'),
    ('ground', (513.5, 7.7), (556.5, 32.5), 'livingroom_shutter2'),
    ('ground', (565, 39.4), (618, 92.3), 'livingroom_shutter3'),
    ('ground', (641, 129.1), (678, 193.4), 'livingroom_shutter4'),
    ('ground', (681.4, 199.1), (717.7, 262), 'livingroom_shutter5'),
    ('first', (149.2, 19.75), (244.4, 19.75), 'bedroom_shutter1'),
    ('first', (297.9, 19.75), (393.1, 19.75), 'bedroom_shutter2'),
    ('first', (675.4, 413.25), (770.5, 413.25), 'office_shutter'),
]
INTERIOR = {'ground': (380, 200), 'first': (420, 220)}  # a point inside the floor, to find the room side of a wall

# small contact sensors on furniture and windows: sensor -> (floor, center, length, 'h' | 'v')
CONTACTS = {
    'kitchen_cupboard1': ('ground', (139, 29), 26, 'v'),
    'kitchen_cupboard2': ('ground', (170, 53.2), 22, 'h'),
    'kitchen_cupboard3': ('ground', (204, 53.2), 22, 'h'),
    'kitchen_cupboard4': ('ground', (238, 53.2), 22, 'h'),
    'kitchen_cupboard5': ('ground', (272, 53.2), 22, 'h'),
    'kitchen_fridge_door': ('ground', (144.6, 140.2), 34, 'v'),
    'livingroom_window1': ('ground', (456, 5.35), 22, 'h'),
    'bedroom_drawer1': ('first', (51.1, 147), 24, 'v'),
    'bedroom_drawer2': ('first', (51.1, 184), 24, 'v'),
    'bathroom_shower_door': ('first', (62.3, 318), 60, 'v'),
    'office_window': ('first', (723, 413.25), 22, 'h'),
}
PRESENCE = {  # sensor -> (floor, center, inner radius, outer radius of the dashed area circles)
    'kitchen_presence': ('ground', (190, 142), 19, 37),
    'livingroom_presence_table': ('ground', (379, 121.5), 19, 37),
    'livingroom_presence_couch': ('ground', (521, 252.4), 19, 37),
    'bedroom_presence': ('first', (263, 168), 24, 46),
    'bathroom_presence': ('first', (116, 326), 13, 26),
    'office_presence': ('first', (640, 292), 24, 46),
}
LIGHTS = {
    'entrance_light1': ('ground', (46, 100)),
    'kitchen_light1': ('ground', (168, 82)),
    'kitchen_light2': ('ground', (262, 128)),
    'livingroom_light1': ('ground', (379, 208)),
    'livingroom_light2': ('ground', (588, 178)),
    'toilet_light': ('ground', (104, 346)),
    'staircase_light': ('ground', (392, 333.5)),
    'walkway_light': ('first', (408, 304)),
    'bathroom_light1': ('first', (80, 350)),
    'bathroom_light2': ('first', (112, 364)),
    'bedroom_light1': ('first', (92, 72)),
    'bedroom_light2': ('first', (432, 122)),
    'bedroom_light3': ('first', (92, 234)),
    'bedroom_light4': ('first', (404, 212)),
    'office_light': ('first', (545, 292)),
}
TVS = {  # sensor -> (floor, x0, y0, x1, y1); the office TV is not on the source plan and sits opposite the couch
    'livingroom_tv_status': ('ground', 617.3, 211.3, 658.8, 291.1),
    'office_tv_status': ('first', 562, 395, 622, 407),
}
SWITCH_STEP = 7.6
SWITCHES = {  # panel -> (floor, top-left button center, {button: (column, row)})
    'entrance_switch': ('ground', (79, 346), {'left': (0, 0)}),
    'kitchen_switch': ('ground', (97, 180), {'top_left': (0, 0), 'top_right': (1, 0), 'bottom_left': (0, 1),
                                           'bottom_right': (1, 1)}),
    'livingroom_switch1': ('ground', (244, 290.5), {'top_left': (0, 0), 'top_right': (1, 0), 'bottom_left': (0, 1)}),
    'livingroom_switch2': ('ground', (394, 17), {'top_left': (0, 0), 'top_right': (1, 0)}),
    'toilet_switch': ('ground', (99, 324), {'left': (0, 0), 'right': (1, 0)}),
    'staircase_switch': ('ground', (180, 290.5), {'left': (0, 0), 'right': (1, 0)}),
    'walkway_switch1': ('first', (448, 353), {'top_left': (0, 0), 'top_right': (1, 0), 'bottom_left': (0, 1),
                                            'bottom_right': (1, 1)}),
    'walkway_switch2': ('first', (190, 276), {'top_left': (0, 0), 'top_right': (1, 0), 'bottom_left': (0, 1),
                                            'bottom_right': (1, 1)}),
    'bathroom_switch': ('first', (161.5, 347), {'top_left': (0, 0), 'top_right': (1, 0), 'bottom_left': (0, 1),
                                              'bottom_right': (1, 1)}),
    'bedroom_switch': ('first', (470.5, 234), {'top_left': (0, 0), 'top_right': (1, 0), 'middle_left': (0, 1),
                                             'middle_right': (1, 1), 'bottom_left': (0, 2),
                                             'bottom_right': (1, 2)}),
    'office_switch': ('first', (475, 352), {'left': (0, 0), 'middle': (1, 0), 'right': (2, 0)}),
}

# furniture: (floor, shape, geometry, label or None[, label center])
FURNITURE = [
    ('ground', 'rect', (91.9, 8.4, 139, 49.2), None),
    ('ground', 'rect', (91.9, 49.2, 141.4, 97.1), 'stove'),
    ('ground', 'rect', (91.9, 116.3, 144.6, 164.2), 'fridge'),
    ('ground', 'rect', (150.2, 8.5, 295.5, 53.2), None),
    ('ground', 'inset', (156, 14, 206, 47.5), 'sink'),
    ('ground', 'rect', (314.7, 79.6, 443.2, 163.4), 'table', (338, 150)),
    ('ground', 'rect', (486.3, 193.7, 555.8, 311.1), None),
    ('ground', 'inset', (486.3, 204, 544, 300.8), 'couch', (514, 219)),
    ('ground', 'rect', (617.3, 211.3, 658.8, 291.1), 'TV', (638, 229)),
    ('ground', 'wc', (119.8, 297.5, 151.8, 361.4), 'WC'),
    ('ground', 'stairs', (171.4, 309.5, 428.9, 361.6), 'Stairs'),
    ('first', 'rect', (7.1, 268, 62.3, 410.3), None),
    ('first', 'inset', (13, 274, 56.5, 404.5), 'shower'),
    ('first', 'rect', (62.3, 372.8, 107, 410.3), None),
    ('first', 'inset', (67, 377.5, 102.5, 406), 'sink'),
    ('first', 'wc', (123.8, 346.4, 155.8, 410.3), 'WC'),
    ('first', 'bed', (182.2, 98.5, 344.5, 261.4), 'bed'),
    ('first', 'stairs', (179.4, 344.8, 409.3, 410.2), 'Stairs'),
    ('first', 'rect', (493.3, 177.8, 606, 234.2), 'desk'),
    ('first', 'rect', (7.1, 125.7, 51.1, 205.6), 'drawers', (27, 165.5)),
    ('first', 'poly', ((735.1, 246.5), (793.9, 348.3), (733.6, 383.1), (674.9, 281.3)), 'couch'),
    ('first', 'rect', (562, 395, 622, 407), None),
]
ROOM_LABELS = [  # (floor, lines, center, font size, rotation)
    ('ground', ['Entrance'], (46, 208), 70, -90),
    ('ground', ['Kitchen'], (232, 216), 78, 0),
    ('ground', ['Living', 'Room'], (512, 100), 78, 0),
    ('ground', ['Toilet'], (135, 279), 44, 0),
    ('first', ['Bedroom'], (265, 62), 78, 0),
    ('first', ['Closet'], (572, 118), 60, 0),
    ('first', ['Bathroom'], (117, 286), 50, 0),
    ('first', ['Walkway'], (298, 312), 62, 0),
    ('first', ['Office'], (585, 350), 78, 0),
]
NOTES = [  # small italic labels: (floor, text, anchor point, text-anchor, font size)
    ('ground', 'Front Door', (98, 386), 'start', 40),
    ('ground', 'cupboards', (221, 71), 'middle', 34),
    ('ground', 'to first floor', (300, 351), 'middle', 28),
    ('first', 'to ground floor', (294, 396), 'middle', 28),
    ('first', 'TV', (592, 388), 'middle', 34),
    ('first', 'closet door', (480, 58), 'end', 30),
]
LEGEND = [('presence', 'presence sensor'), ('contact', 'door, cupboard, drawer or window sensor'),
          ('switch', 'wall switch (one square per button)'), ('light', 'light'), ('shutter', 'window shutter'),
          ('tv', 'TV (on / off)')]


def px(floor, x, y):
    return M + S * x, OY[floor] + S * y


def fmt(v):
    return f'{v:.1f}'.rstrip('0').rstrip('.')


def pts(floor, points):
    return ' '.join(f'{fmt(a)},{fmt(b)}' for a, b in (px(floor, x, y) for x, y in points))


def text(x, y, s, size, fill, weight=400, italic=False, anchor='middle', rotate=0, halo=0):
    style = f'font-size:{size}px;font-weight:{weight};fill:{fill}' + (';font-style:italic' if italic else '')
    if halo:
        style += f';stroke:#fff;stroke-width:{halo}px;stroke-linejoin:round;paint-order:stroke'
    tr = f' transform="rotate({rotate} {fmt(x)} {fmt(y)})"' if rotate else ''
    return (f'<text x="{fmt(x)}" y="{fmt(y)}" text-anchor="{anchor}" dominant-baseline="central" style="{style}"{tr}>'
            f'{s}</text>')


def bar(floor, a, b, thick, fill, stroke=None, sw=0, r=0):
    """Rectangle of `thick` pixels centered on the segment a-b (points)."""
    (x0, y0), (x1, y1) = px(floor, *a), px(floor, *b)
    length, ang = math.hypot(x1 - x0, y1 - y0), math.degrees(math.atan2(y1 - y0, x1 - x0))
    st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ''
    return (f'<rect x="{fmt(-length / 2)}" y="{fmt(-thick / 2)}" width="{fmt(length)}" height="{fmt(thick)}" rx="{r}" '
            f'fill="{fill}"{st} transform="translate({fmt((x0 + x1) / 2)} {fmt((y0 + y1) / 2)}) rotate({fmt(ang)})"/>')


def inward(floor, a, b, dist):
    """Segment a-b moved `dist` points toward the inside of the floor."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = (-dy / math.hypot(dx, dy), dx / math.hypot(dx, dy))
    mid, ref = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), INTERIOR[floor]
    if n[0] * (ref[0] - mid[0]) + n[1] * (ref[1] - mid[1]) < 0:
        n = (-n[0], -n[1])
    return tuple((p[0] + n[0] * dist, p[1] + n[1] * dist) for p in (a, b))


def shrink(a, b, frac):
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    return tuple((mid[0] + (p[0] - mid[0]) * frac, mid[1] + (p[1] - mid[1]) * frac) for p in (a, b))


def shutter_bars():
    """shutter sensor -> (floor, end, end) of its bar, inside the window."""
    return {sid: (floor, *shrink(*inward(floor, a, b, 7.5), 0.62)) for floor, a, b, sid in WINDOWS if sid}


def switch_buttons():
    out = {}
    for panel, (floor, (x, y), buttons) in SWITCHES.items():
        for name, (col, row) in buttons.items():
            out[f'{panel}_{name}'] = (floor, (x + col * SWITCH_STEP, y + row * SWITCH_STEP))
    return out


def sensor_points():
    """sensor -> (floor, x, y) in PDF points, for all 82 binary sources."""
    out = {s: (f, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for s, (f, a, b) in SENSOR_DOORS.items()}
    out.update({s: (f, *c) for s, (f, c, _, _) in CONTACTS.items()})
    out.update({s: (f, *c) for s, (f, c, _, _) in PRESENCE.items()})
    out.update({s: (f, *c) for s, (f, c) in LIGHTS.items()})
    out.update({s: (f, (x0 + x1) / 2, (y0 + y1) / 2) for s, (f, x0, y0, x1, y1) in TVS.items()})
    out.update({s: (f, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for s, (f, a, b) in shutter_bars().items()})
    out.update({s: (f, *c) for s, (f, c) in switch_buttons().items()})
    return out


def sensor_pixels():
    """sensor -> [x, y] in image pixels, y from the top."""
    return {s: [round(v) for v in px(f, x, y)] for s, (f, x, y) in sorted(sensor_points().items())}


# ------------------------------------------------------------------ markers

def dot(x, y):
    return f'<circle cx="{fmt(x)}" cy="{fmt(y)}" r="15" fill="url(#dot)"/>'


def light(x, y, r=16):
    d = r * 0.7071
    return (f'<g stroke="{TEXT}" stroke-width="3" fill="#fff"><circle cx="{fmt(x)}" cy="{fmt(y)}" r="{r}"/>'
            f'<path d="M{fmt(x - d)} {fmt(y - d)}L{fmt(x + d)} {fmt(y + d)}M{fmt(x - d)} {fmt(y + d)}L{fmt(x + d)} {fmt(y - d)}"/></g>')


def button(x, y, size=19):
    return (f'<rect x="{fmt(x - size / 2)}" y="{fmt(y - size / 2)}" width="{size}" height="{size}" rx="2" fill="#fff" '
            f'stroke="{TITLE}" stroke-width="3"/>')


def furniture(floor, shape, geom, label, at=None):
    out = []
    if shape in ('rect', 'inset', 'bed', 'stairs'):
        x0, y0 = px(floor, geom[0], geom[1])
        x1, y1 = px(floor, geom[2], geom[3])
        w, h = x1 - x0, y1 - y0
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if shape == 'rect':
        out.append(f'<rect x="{fmt(x0)}" y="{fmt(y0)}" width="{fmt(w)}" height="{fmt(h)}" fill="{FURN}"/>')
    elif shape == 'inset':
        out.append(f'<rect x="{fmt(x0)}" y="{fmt(y0)}" width="{fmt(w)}" height="{fmt(h)}" rx="14" fill="{FURN_LIGHT}"/>')
    elif shape == 'bed':
        out.append(f'<rect x="{fmt(x0)}" y="{fmt(y0)}" width="{fmt(w)}" height="{fmt(h)}" fill="{FURN}"/>')
        for k in (0, 1):
            out.append(f'<rect x="{fmt(x0 + 40 + k * (w - 80 + 30) / 2)}" y="{fmt(y1 - 118)}" width="{fmt((w - 80 - 30) / 2)}" '
                       f'height="84" rx="16" fill="{PILLOW}"/>')
        cy -= 150
    elif shape == 'stairs':
        out.append(f'<rect x="{fmt(x0)}" y="{fmt(y0)}" width="{fmt(w)}" height="{fmt(h)}" fill="{FURN_LIGHT}"/>')
        n = round(w / 44)
        for k in range(n):
            out.append(f'<rect x="{fmt(x0 + (k + 0.5) * w / n - 12)}" y="{fmt(y0)}" width="24" height="{fmt(h)}" fill="{FURN}" '
                       f'opacity="0.55"/>')
        cy -= 26
    elif shape == 'wc':
        x0, y0 = px(floor, geom[0], geom[1])
        x1, y1 = px(floor, geom[2], geom[3])
        cx, w = (x0 + x1) / 2, x1 - x0
        out.append(f'<rect x="{fmt(x0 + 8)}" y="{fmt(y1 - 52)}" width="{fmt(w - 16)}" height="46" rx="6" fill="{FURN}"/>')
        out.append(f'<ellipse cx="{fmt(cx)}" cy="{fmt(y0 + 78)}" rx="{fmt(w / 2)}" ry="74" fill="{FURN}"/>')
        cy = y0 + 78
    elif shape == 'poly':
        out.append(f'<polygon points="{pts(floor, geom)}" fill="{FURN}"/>')
        cx = sum(px(floor, *p)[0] for p in geom) / len(geom)
        cy = sum(px(floor, *p)[1] for p in geom) / len(geom)
    if label:
        if at:
            cx, cy = px(floor, *at)
        rot = 60 if shape == 'poly' else (-90 if shape == 'rect' and h > 2.2 * w else 0)
        size = 40 if shape == 'stairs' else (30 if label == 'drawers' else 34)
        out.append(text(cx, cy, label, size, TEXT, italic=True, rotate=rot))
    return ''.join(out)


def legend_icon(kind, x, y):
    if kind == 'presence':
        return dot(x, y)
    if kind == 'contact':
        return f'<rect x="{x - 26}" y="{y - 9}" width="52" height="18" fill="{DOOR}" stroke="{INK}" stroke-width="3"/>'
    if kind == 'switch':
        return button(x - 11.5, y) + button(x + 11.5, y)
    if kind == 'light':
        return light(x, y)
    if kind == 'shutter':
        return f'<rect x="{x - 26}" y="{y - 6}" width="52" height="12" rx="6" fill="{SHUTTER}"/>'
    return f'<rect x="{x - 26}" y="{y - 10}" width="52" height="20" fill="{FURN}"/>' + dot(x, y)


def build_svg():
    layers = {k: [] for k in ('furniture', 'areas', 'walls', 'openings', 'markers', 'labels')}
    for item in FURNITURE:
        layers['furniture'].append(furniture(*item))
    for floor, lines in WALLS.items():
        for line in lines:
            closed = line[0] == line[-1] and len(line) > 2
            d = 'M' + ' L'.join(pts(floor, [p]) for p in (line[:-1] if closed else line)) + (' Z' if closed else '')
            layers['walls'].append(f'<path d="{d}" fill="none" stroke="{INK}" stroke-width="{WALL}" '
                                   f'stroke-linecap="square" stroke-linejoin="miter"/>')
    # openings: white gap, then the door / window symbol
    for floor, a, b, hinge, closed, opened in PLAIN_DOORS:
        layers['openings'].append(bar(floor, a, b, WALL + 6, '#fff'))
        ang = math.radians(24)
        tip = tuple(hinge[i] + (closed[i] - hinge[i]) * math.cos(ang) + (opened[i] - hinge[i]) * math.sin(ang)
                    for i in (0, 1))
        layers['openings'].append(f'<path d="M{pts(floor, [hinge])} L{pts(floor, [tip])}" stroke="{INK}" '
                                  f'stroke-width="4" stroke-linecap="round"/>')
    for floor, a, b, _ in WINDOWS:
        layers['openings'].append(bar(floor, a, b, WALL + 6, '#fff'))
        layers['openings'].append(bar(floor, a, b, WALL + 3, '#fff', INK, 2.5))
        layers['openings'].append(bar(floor, a, b, 2.5, INK))
    for sid, (floor, a, b) in shutter_bars().items():
        layers['markers'].append(bar(floor, a, b, 12, SHUTTER, r=6))
    for sid, (floor, a, b) in SENSOR_DOORS.items():
        layers['openings'].append(bar(floor, a, b, WALL + 6, '#fff'))
        layers['markers'].append(bar(floor, a, b, 26, DOOR, INK, 3))
    for sid, (floor, c, length, o) in CONTACTS.items():
        half = (length / 2, 0) if o == 'h' else (0, length / 2)
        layers['markers'].append(bar(floor, (c[0] - half[0], c[1] - half[1]), (c[0] + half[0], c[1] + half[1]), 16,
                                     DOOR, INK, 3))
    for sid, (floor, c, r0, r1) in PRESENCE.items():
        x, y = px(floor, *c)
        layers['areas'].append(f'<circle cx="{fmt(x)}" cy="{fmt(y)}" r="{fmt(r0 * S)}" fill="none" stroke="{DASH}" '
                               f'stroke-width="3" stroke-dasharray="14 12"/>')
        layers['areas'].append(f'<circle cx="{fmt(x)}" cy="{fmt(y)}" r="{fmt(r1 * S)}" fill="none" stroke="{FURN}" '
                               f'stroke-width="3" stroke-dasharray="7 9"/>')
        layers['markers'].append(dot(x, y))
        if r0 > 15:
            layers['labels'].append(text(x + 24, y + 22, 'presence', 27, TEXT, italic=True, anchor='start'))
            layers['labels'].append(text(x + 24, y + 52, 'sensor', 27, TEXT, italic=True, anchor='start'))
    for sid, (floor, c) in LIGHTS.items():
        layers['markers'].append(light(*px(floor, *c)))
    for sid, (floor, c) in switch_buttons().items():
        layers['markers'].append(button(*px(floor, *c)))
    for sid, (floor, x0, y0, x1, y1) in TVS.items():
        layers['markers'].append(dot(*px(floor, (x0 + x1) / 2, (y0 + y1) / 2)))
    for floor, lines, (cx, cy), size, rot in ROOM_LABELS:
        x, y = px(floor, cx, cy)
        for k, line in enumerate(lines):
            layers['labels'].append(text(x, y + (k - (len(lines) - 1) / 2) * size * 1.12, line, size, ROOM, 700,
                                         rotate=rot, halo=7))
    for floor, s, (cx, cy), anchor, size in NOTES:
        layers['labels'].append(text(*px(floor, cx, cy), s, size, TEXT, italic=True, anchor=anchor))
    for floor, title in (('first', 'First Floor'), ('ground', 'Ground Floor')):
        layers['labels'].append(text(M, OY[floor] - 62, title, 78, TITLE, 700, anchor='start'))
    lx, ly = px('ground', 668, 4)
    for k, (kind, label) in enumerate(LEGEND):
        layers['labels'].append(legend_icon(kind, lx, ly + k * 52))
        layers['labels'].append(text(lx + 48, ly + k * 52, label, 30, TEXT, italic=True, anchor='start'))
    layers['labels'].append(text(W - M, H - 86, 'Orange4Home Apartment Plan - Amiqual4Home', 34, ROOM, italic=True,
                                 anchor='end'))
    layers['labels'].append(text(W - M, H - 44, 're-rendered from Cumin et al. (2017)', 24, ROOM, italic=True,
                                 anchor='end'))
    defs = ('<defs><radialGradient id="dot"><stop offset="0" stop-color="#000"/><stop offset="0.42" stop-color="#111"/>'
            '<stop offset="0.6" stop-color="#808080" stop-opacity="0.75"/><stop offset="1" stop-color="#bfbfbf" '
            'stop-opacity="0"/></radialGradient></defs>')
    body = ''.join(f'<g id="{k}">{"".join(v)}</g>' for k, v in layers.items())
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
            f'font-family="{FONT}">{defs}<rect width="{W}" height="{H}" fill="#fff"/>{body}</svg>')


# ------------------------------------------------------------------ output

def font_css():
    """@font-face rules for Inter (the typeface of the CASAS re-renders), from a local cache or the Google Fonts
    repository; without them the browser falls back to Helvetica Neue."""
    css = []
    for style, (name, remote) in FONTS.items():
        path = FONT_CACHE / name
        if not path.exists():
            try:
                FONT_CACHE.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(FONT_URL + remote, path)
            except OSError as e:
                print(f'Inter ({style}) not available, falling back to Helvetica Neue: {e}')
                continue
        css.append(f"@font-face{{font-family:Inter;font-style:{style};font-weight:100 900;"
                   f"src:url(data:font/ttf;base64,{base64.b64encode(path.read_bytes()).decode()})}}")
    return ''.join(css)


def render_png(svg: str, path: Path):
    with tempfile.TemporaryDirectory() as tmp:
        html = Path(tmp) / 'plan.html'
        html.write_text(f'<html><head><style>{font_css()}html,body{{margin:0;background:#fff}}svg{{display:block}}'
                        f'</style></head><body>{svg}</body></html>')
        subprocess.run([CHROME, '--headless=new', '--disable-gpu', '--hide-scrollbars',
                        '--force-device-scale-factor=1', f'--window-size={W},{H}', '--virtual-time-budget=3000',
                        f'--screenshot={path}', html.as_uri()], check=True, capture_output=True)


def update_metadata(pixels: dict):
    """Write `sensor_coordinates` (x, y from the bottom of the image, as for the CASAS houses) into the orange
    entry, leaving the rest of the hand-formatted file untouched."""
    raw = METADATA.read_text()
    start = raw.index('"orange": {')
    head, entry = raw[:start], raw[start:]
    lines = ',\n'.join(f'      "{s}": [{x}, {H - y}]' for s, (x, y) in pixels.items())
    block = f'"sensor_coordinates": {{\n{lines}\n    }}'
    if '"sensor_coordinates"' in entry:
        entry = re.sub(r'"sensor_coordinates": \{.*?\n    \}', block, entry, count=1, flags=re.S)
    else:
        entry = re.sub(r'\n  \}\n\}\s*$', f',\n    {block}\n  }}\n}}\n', entry, count=1)
    METADATA.write_text(head + entry)
    meta = json.loads(METADATA.read_text())['orange']
    missing = set(meta['sensor_location']) ^ set(meta['sensor_coordinates'])
    assert not missing, f'sensors without coordinates or without metadata: {sorted(missing)}'


def main():
    svg = build_svg()
    (FLOOR_PLANS / 'orange.svg').write_text(svg)
    render_png(svg, FLOOR_PLANS / 'orange.png')
    pixels = sensor_pixels()
    update_metadata(pixels)
    print(f'{FLOOR_PLANS / "orange.png"}  {W}x{H} px, {len(pixels)} sensors')


if __name__ == '__main__':
    main()
