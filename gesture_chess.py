"""
gesture_chess.py

Gesture-controlled virtual chess using:
- OpenCV
- MediaPipe Hand Landmarker
- python-chess

Controls
--------
Right hand:
    Index finger       -> hover a square
    Thumb + index pinch -> select a piece
    Move while pinched -> drag selected piece
    Release pinch      -> attempt the move

Left hand:
    Open palm + movement -> orbit the board
    Fist                 -> freeze board view

Keyboard:
    R       Reset game
    U       Undo last move
    F       Flip board
    1       White promotion to Queen
    2       White promotion to Rook
    3       White promotion to Bishop
    4       White promotion to Knight
    Q / ESC Quit

Mouse fallback:
    Left click square -> select
    Left click another square -> move

Requires:
    hand_landmarker.task
"""

import os
import sys
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import chess
from dotenv import load_dotenv

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# Configuration
# ============================================================

load_dotenv()

W_FRAME = 1280
H_FRAME = 720

CAMERA_INDEX = 0

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hand_landmarker.task",
)

WINDOW_NAME = "Gesture Chess"

# Board local dimensions
BOARD_SIZE = 8.0
SQUARE_SIZE = BOARD_SIZE / 8.0

# Board position in the virtual 3D scene
BOARD_CENTER_X = 0.0
BOARD_CENTER_Y = 0.0
BOARD_Z = 0.0

# Board thickness
BOARD_THICKNESS = 0.35

# Camera / projection
CX = 650
CY = 390
FOCAL = 72.0

# Default board orientation
ROT_X = 58.0
ROT_Y = -18.0
ROT_Z = 0.0

# Board orbit sensitivity
ORBIT_SENSITIVITY_X = 0.22
ORBIT_SENSITIVITY_Y = 0.18

# Gesture thresholds
PINCH_ON = 0.34
PINCH_OFF = 0.48

PINCH_STABLE_FRAMES = 2
RELEASE_STABLE_FRAMES = 2

# Smooth fingertip movement
FINGER_EMA = 0.42

# Maximum distance between hands for association
HAND_MATCH_DISTANCE = 180

# Gesture trail
TRAIL_LENGTH = 12

# Board hover margin
BOARD_HOVER_MARGIN = 0.08

# UI
HUD_X = 28
HUD_Y = 32

# Colors
WHITE = (245, 245, 245)
BLACK = (20, 20, 20)
GRAY = (150, 150, 150)
DARK_GRAY = (45, 45, 45)

BOARD_LIGHT = (238, 218, 187)
BOARD_DARK = (116, 81, 53)

GREEN = (60, 220, 100)
BLUE = (80, 180, 255)
YELLOW = (0, 220, 255)
RED = (60, 70, 230)

WHITE_PIECE = (245, 245, 240)
BLACK_PIECE = (45, 45, 48)

OUTLINE = (30, 30, 30)


# ============================================================
# Math / projection
# ============================================================

def Rx(angle):
    angle = np.radians(angle)
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array(
        [
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ],
        dtype=np.float64,
    )


def Ry(angle):
    angle = np.radians(angle)
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array(
        [
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c],
        ],
        dtype=np.float64,
    )


def Rz(angle):
    angle = np.radians(angle)
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array(
        [
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )


def rotation_matrix():
    return Rz(ROT_Z) @ Ry(ROT_Y) @ Rx(ROT_X)


def project_point(point):
    """
    Project one 3D point into screen space.

    The renderer uses an orthographic-style projection with
    depth used only for painter ordering.
    """
    x, y, z = point

    return (
        int(CX + x * FOCAL),
        int(CY - y * FOCAL),
    )


def project_points(points):
    return [project_point(p) for p in points]


def transform_point(point):
    return rotation_matrix() @ np.asarray(point, dtype=np.float64)


def world_to_screen(point):
    return project_point(transform_point(point))


# ============================================================
# Board geometry
# ============================================================

def square_corners(file_index, rank_index, z=0.0):
    """
    Return the four local 3D corners of a chess square.

    file_index:
        0 = a ... 7 = h

    rank_index:
        0 = rank 1 ... 7 = rank 8
    """

    x0 = -BOARD_SIZE / 2 + file_index * SQUARE_SIZE
    x1 = x0 + SQUARE_SIZE

    y0 = -BOARD_SIZE / 2 + rank_index * SQUARE_SIZE
    y1 = y0 + SQUARE_SIZE

    return [
        (x0, y0, z),
        (x1, y0, z),
        (x1, y1, z),
        (x0, y1, z),
    ]


def square_center(file_index, rank_index, z=0.0):
    x = -BOARD_SIZE / 2 + (file_index + 0.5) * SQUARE_SIZE
    y = -BOARD_SIZE / 2 + (rank_index + 0.5) * SQUARE_SIZE

    return (x, y, z)


def draw_polygon(frame, points, fill, outline=None, thickness=1):
    pts = np.asarray(points, dtype=np.int32)

    cv2.fillConvexPoly(frame, pts, fill)

    if outline is not None:
        cv2.polylines(
            frame,
            [pts],
            True,
            outline,
            thickness,
            cv2.LINE_AA,
        )


def draw_board(frame, board_flipped=False):
    """
    Draw the 3D chess board and return projected square polygons.
    """

    square_polygons = {}

    # Board thickness / side walls and premium frame
    board_shadow = [
        (-BOARD_SIZE / 2 - 0.25, -BOARD_SIZE / 2 - 0.28, BOARD_Z - 0.10),
        (BOARD_SIZE / 2 + 0.25, -BOARD_SIZE / 2 - 0.28, BOARD_Z - 0.10),
        (BOARD_SIZE / 2 + 0.30, BOARD_SIZE / 2 + 0.32, BOARD_Z - 0.10),
        (-BOARD_SIZE / 2 - 0.30, BOARD_SIZE / 2 + 0.32, BOARD_Z - 0.10),
    ]

    shadow_screen = project_points(
        [transform_point(p) for p in board_shadow]
    )

    shadow_overlay = frame.copy()
    draw_polygon(
        shadow_overlay,
        shadow_screen,
        (18, 15, 12),
        None,
        1,
    )
    cv2.addWeighted(
        shadow_overlay,
        0.30,
        frame,
        0.70,
        0,
        frame,
    )

    frame_corners = [
        (-BOARD_SIZE / 2 - 0.35, -BOARD_SIZE / 2 - 0.35, BOARD_Z + 0.02),
        (BOARD_SIZE / 2 + 0.35, -BOARD_SIZE / 2 - 0.35, BOARD_Z + 0.02),
        (BOARD_SIZE / 2 + 0.35, BOARD_SIZE / 2 + 0.35, BOARD_Z + 0.02),
        (-BOARD_SIZE / 2 - 0.35, BOARD_SIZE / 2 + 0.35, BOARD_Z + 0.02),
    ]

    frame_screen = project_points(
        [transform_point(p) for p in frame_corners]
    )

    draw_polygon(
        frame,
        frame_screen,
        (91, 73, 58),
        (144, 112, 78),
        6,
    )

    top_corners = [
        (-BOARD_SIZE / 2, -BOARD_SIZE / 2, BOARD_Z),
        (BOARD_SIZE / 2, -BOARD_SIZE / 2, BOARD_Z),
        (BOARD_SIZE / 2, BOARD_SIZE / 2, BOARD_Z),
        (-BOARD_SIZE / 2, BOARD_SIZE / 2, BOARD_Z),
    ]

    bottom_corners = [
        (-BOARD_SIZE / 2, -BOARD_SIZE / 2, BOARD_Z - BOARD_THICKNESS),
        (BOARD_SIZE / 2, -BOARD_SIZE / 2, BOARD_Z - BOARD_THICKNESS),
        (BOARD_SIZE / 2, BOARD_SIZE / 2, BOARD_Z - BOARD_THICKNESS),
        (-BOARD_SIZE / 2, BOARD_SIZE / 2, BOARD_Z - BOARD_THICKNESS),
    ]

    top_screen = project_points(
        [transform_point(p) for p in top_corners]
    )

    bottom_screen = project_points(
        [transform_point(p) for p in bottom_corners]
    )

    # Side walls
    draw_polygon(
        frame,
        [
            top_screen[0],
            top_screen[1],
            bottom_screen[1],
            bottom_screen[0],
        ],
        (55, 40, 30),
        BLACK,
        2,
    )

    draw_polygon(
        frame,
        [
            top_screen[1],
            top_screen[2],
            bottom_screen[2],
            bottom_screen[1],
        ],
        (70, 48, 34),
        BLACK,
        2,
    )

    draw_polygon(
        frame,
        [
            top_screen[2],
            top_screen[3],
            bottom_screen[3],
            bottom_screen[2],
        ],
        (50, 36, 27),
        BLACK,
        2,
    )

    draw_polygon(
        frame,
        [
            top_screen[3],
            top_screen[0],
            bottom_screen[0],
            bottom_screen[3],
        ],
        (65, 45, 32),
        BLACK,
        2,
    )

    board_outline = np.asarray(top_screen, dtype=np.int32)
    cv2.polylines(
        frame,
        [board_outline],
        True,
        (85, 72, 62),
        4,
        cv2.LINE_AA,
    )

    # Squares

    for rank in range(8):
        for file_index in range(8):

            display_file = 7 - file_index if board_flipped else file_index
            display_rank = 7 - rank if board_flipped else rank

            corners = square_corners(
                display_file,
                display_rank,
                BOARD_Z,
            )

            screen = project_points(
                [transform_point(p) for p in corners]
            )

            is_light = (
                (file_index + rank) % 2 == 0
            )

            color = BOARD_LIGHT if is_light else BOARD_DARK

            square_name = chess.square_name(
                chess.square(
                    file_index,
                    rank,
                )
            )

            square_polygons[square_name] = screen

            draw_polygon(
                frame,
                screen,
                color,
                (35, 35, 35),
                1,
            )

    # Board labels

    for file_index in range(8):
        center = square_center(
            file_index,
            0,
            BOARD_Z + 0.01,
        )

        sx, sy = world_to_screen(center)

        label_file = (
            chess.FILE_NAMES[
                7 - file_index
                if board_flipped
                else file_index
            ]
        )

        cv2.putText(
            frame,
            label_file,
            (sx - 5, sy + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            BLACK,
            1,
            cv2.LINE_AA,
        )

    for rank_index in range(8):
        center = square_center(
            0,
            rank_index,
            BOARD_Z + 0.01,
        )

        sx, sy = world_to_screen(center)

        label_rank = str(
            8 - rank_index
            if board_flipped
            else rank_index + 1
        )

        cv2.putText(
            frame,
            label_rank,
            (sx - 28, sy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            BLACK,
            1,
            cv2.LINE_AA,
        )

    return square_polygons


# ============================================================
# Chess rendering
# ============================================================

PIECE_SYMBOLS = {
    chess.PAWN: "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK: "R",
    chess.QUEEN: "Q",
    chess.KING: "K",
}


def draw_piece(
    frame,
    piece,
    square,
    board_flipped=False,
    selected=False,
):
    """
    Procedurally render a recognizable chess piece.

    This avoids relying on Unicode chess glyph support.
    """

    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)

    if board_flipped:
        file_index = 7 - file_index
        rank_index = 7 - rank_index

    center3d = square_center(
        file_index,
        rank_index,
        BOARD_Z + 0.20,
    )

    sx, sy = world_to_screen(center3d)

    # Approximate projected square width.
    c1 = world_to_screen(
        square_center(
            file_index,
            rank_index,
            BOARD_Z,
        )
    )

    c2 = world_to_screen(
        square_center(
            min(file_index + 1, 7),
            rank_index,
            BOARD_Z,
        )
    )

    radius = max(
        10,
        int(abs(c2[0] - c1[0]) * 0.30),
    )

    fill = (
        WHITE_PIECE
        if piece.color == chess.WHITE
        else BLACK_PIECE
    )

    outline = (
        BLUE
        if selected
        else OUTLINE
    )

    ptype = piece.piece_type

    # Base

    base_y = sy + int(radius * 0.55)

    cv2.ellipse(
        frame,
        (sx, base_y),
        (radius, max(5, radius // 3)),
        0,
        0,
        360,
        outline,
        -1,
        cv2.LINE_AA,
    )

    cv2.ellipse(
        frame,
        (sx, base_y),
        (max(3, radius - 2), max(3, radius // 3 - 2)),
        0,
        0,
        360,
        fill,
        -1,
        cv2.LINE_AA,
    )

    # Pawn

    if ptype == chess.PAWN:

        cv2.circle(
            frame,
            (sx, sy - radius // 3),
            max(5, radius // 3),
            fill,
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            frame,
            (sx, sy - radius // 3),
            max(5, radius // 3),
            outline,
            2,
            cv2.LINE_AA,
        )

        body = np.array(
            [
                [sx - radius // 2, sy + radius // 2],
                [sx + radius // 2, sy + radius // 2],
                [sx + radius // 3, sy - radius // 4],
                [sx - radius // 3, sy - radius // 4],
            ],
            dtype=np.int32,
        )

        cv2.fillConvexPoly(
            frame,
            body,
            fill,
        )

        cv2.polylines(
            frame,
            [body],
            True,
            outline,
            2,
            cv2.LINE_AA,
        )

    # Rook

    elif ptype == chess.ROOK:

        body = np.array(
            [
                [sx - radius // 2, sy + radius // 2],
                [sx + radius // 2, sy + radius // 2],
                [sx + radius // 3, sy - radius // 2],
                [sx - radius // 3, sy - radius // 2],
            ],
            dtype=np.int32,
        )

        cv2.fillConvexPoly(
            frame,
            body,
            fill,
        )

        cv2.polylines(
            frame,
            [body],
            True,
            outline,
            2,
            cv2.LINE_AA,
        )

        top = cv2.boxPoints(
            (
                (sx, sy - radius // 2),
                (radius // 2, max(4, radius // 3)),
                0,
            )
        ).astype(np.int32)

        cv2.fillConvexPoly(
            frame,
            top,
            fill,
        )

        cv2.polylines(
            frame,
            [top],
            True,
            outline,
            2,
            cv2.LINE_AA,
        )

        # Battlements
        for dx in (-radius // 3, 0, radius // 3):
            cv2.rectangle(
                frame,
                (
                    sx + dx - max(2, radius // 10),
                    sy - radius // 2 - radius // 5,
                ),
                (
                    sx + dx + max(2, radius // 10),
                    sy - radius // 2,
                ),
                fill,
                -1,
            )

    # Knight

    elif ptype == chess.KNIGHT:

        points = np.array(
            [
                [sx - radius // 2, sy + radius // 2],
                [sx + radius // 2, sy + radius // 2],
                [sx + radius // 3, sy],
                [sx + radius // 5, sy - radius // 2],
                [sx - radius // 4, sy - radius // 3],
                [sx - radius // 2, sy],
            ],
            dtype=np.int32,
        )

        cv2.fillConvexPoly(
            frame,
            points,
            fill,
        )

        cv2.polylines(
            frame,
            [points],
            True,
            outline,
            2,
            cv2.LINE_AA,
        )

        # Horse ear
        cv2.line(
            frame,
            (
                sx + radius // 8,
                sy - radius // 2,
            ),
            (
                sx + radius // 4,
                sy - radius * 3 // 4,
            ),
            outline,
            3,
            cv2.LINE_AA,
        )

    # Bishop

    elif ptype == chess.BISHOP:

        cv2.ellipse(
            frame,
            (sx, sy - radius // 4),
            (
                max(5, radius // 2),
                max(8, radius * 3 // 4),
            ),
            0,
            0,
            360,
            fill,
            -1,
            cv2.LINE_AA,
        )

        cv2.ellipse(
            frame,
            (sx, sy - radius // 4),
            (
                max(5, radius // 2),
                max(8, radius * 3 // 4),
            ),
            0,
            0,
            360,
            outline,
            2,
            cv2.LINE_AA,
        )

        cv2.line(
            frame,
            (
                sx - radius // 4,
                sy - radius // 2,
            ),
            (
                sx + radius // 4,
                sy,
            ),
            outline,
            2,
            cv2.LINE_AA,
        )

    # Queen

    elif ptype == chess.QUEEN:

        cv2.circle(
            frame,
            (sx, sy - radius // 3),
            max(7, radius // 2),
            fill,
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            frame,
            (sx, sy - radius // 3),
            max(7, radius // 2),
            outline,
            2,
            cv2.LINE_AA,
        )

        for angle in np.linspace(
            -65,
            65,
            5,
        ):
            rad = np.radians(angle)

            x2 = int(
                sx
                + np.sin(rad) * radius * 0.55
            )

            y2 = int(
                sy
                - radius // 3
                - np.cos(rad) * radius * 0.55
            )

            cv2.line(
                frame,
                (sx, sy - radius // 3),
                (x2, y2),
                outline,
                2,
                cv2.LINE_AA,
            )

    # King

    elif ptype == chess.KING:

        body = np.array(
            [
                [sx - radius // 2, sy + radius // 2],
                [sx + radius // 2, sy + radius // 2],
                [sx + radius // 3, sy - radius // 5],
                [sx - radius // 3, sy - radius // 5],
            ],
            dtype=np.int32,
        )

        cv2.fillConvexPoly(
            frame,
            body,
            fill,
        )

        cv2.polylines(
            frame,
            [body],
            True,
            outline,
            2,
            cv2.LINE_AA,
        )

        # Cross
        cross_w = max(3, radius // 8)

        cv2.rectangle(
            frame,
            (
                sx - cross_w,
                sy - radius * 3 // 4,
            ),
            (
                sx + cross_w,
                sy - radius // 4,
            ),
            fill,
            -1,
        )

        cv2.rectangle(
            frame,
            (
                sx - radius // 4,
                sy - radius // 2,
            ),
            (
                sx + radius // 4,
                sy - radius // 2 + 2 * cross_w,
            ),
            fill,
            -1,
        )

        cv2.rectangle(
            frame,
            (
                sx - cross_w,
                sy - radius * 3 // 4,
            ),
            (
                sx + cross_w,
                sy - radius // 4,
            ),
            outline,
            1,
        )

    # Small piece identifier helps distinguish silhouettes.
    symbol = PIECE_SYMBOLS[ptype]

    cv2.putText(
        frame,
        symbol,
        (
            sx - 5,
            sy + radius + 18,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        outline,
        1,
        cv2.LINE_AA,
    )


def draw_pieces(
    frame,
    board,
    selected_square=None,
    board_flipped=False,
):
    """
    Draw pieces back-to-front using projected depth.
    """

    pieces = []

    for square, piece in board.piece_map().items():

        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)

        if board_flipped:
            file_index = 7 - file_index
            rank_index = 7 - rank_index

        center = transform_point(
            square_center(
                file_index,
                rank_index,
                BOARD_Z + 0.2,
            )
        )

        pieces.append(
            (
                center[2],
                square,
                piece,
            )
        )

    pieces.sort(
        key=lambda item: item[0]
    )

    for _, square, piece in pieces:

        draw_piece(
            frame,
            piece,
            square,
            board_flipped,
            selected_square == square,
        )


# ============================================================
# Board / screen mapping
# ============================================================

def get_board_corners_screen(board_flipped=False):
    """
    Return projected corners in the order:

        top-left
        top-right
        bottom-right
        bottom-left

    from the current visual board orientation.
    """

    if not board_flipped:
        corners = [
            (-BOARD_SIZE / 2, BOARD_SIZE / 2, BOARD_Z),
            (BOARD_SIZE / 2, BOARD_SIZE / 2, BOARD_Z),
            (BOARD_SIZE / 2, -BOARD_SIZE / 2, BOARD_Z),
            (-BOARD_SIZE / 2, -BOARD_SIZE / 2, BOARD_Z),
        ]
    else:
        corners = [
            (BOARD_SIZE / 2, -BOARD_SIZE / 2, BOARD_Z),
            (-BOARD_SIZE / 2, -BOARD_SIZE / 2, BOARD_Z),
            (-BOARD_SIZE / 2, BOARD_SIZE / 2, BOARD_Z),
            (BOARD_SIZE / 2, BOARD_SIZE / 2, BOARD_Z),
        ]

    return np.asarray(
        [
            world_to_screen(c)
            for c in corners
        ],
        dtype=np.float32,
    )


def screen_to_board_uv(point, board_flipped=False):
    """
    Map screen coordinates to normalized board coordinates.

    Returns:
        u, v in [0, 1]
        or None if outside the board.
    """

    corners = get_board_corners_screen(
        board_flipped
    )

    src = corners.astype(np.float32)

    dst = np.array(
        [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(
        src,
        dst,
    )

    point = np.array(
        [[[float(point[0]), float(point[1])]]],
        dtype=np.float32,
    )

    mapped = cv2.perspectiveTransform(
        point,
        matrix,
    )[0][0]

    u, v = float(mapped[0]), float(mapped[1])

    if (
        u < -BOARD_HOVER_MARGIN
        or u > 1 + BOARD_HOVER_MARGIN
        or v < -BOARD_HOVER_MARGIN
        or v > 1 + BOARD_HOVER_MARGIN
    ):
        return None

    return (
        max(0.0, min(1.0, u)),
        max(0.0, min(1.0, v)),
    )


def screen_to_square(
    point,
    board_flipped=False,
):
    uv = screen_to_board_uv(
        point,
        board_flipped,
    )

    if uv is None:
        return None

    u, v = uv

    file_index = min(
        7,
        max(0, int(u * 8)),
    )

    rank_from_top = min(
        7,
        max(0, int(v * 8)),
    )

    if board_flipped:
        file_index = 7 - file_index
        rank_index = rank_from_top
    else:
        file_index = file_index
        rank_index = 7 - rank_from_top

    return chess.square(
        file_index,
        rank_index,
    )


# ============================================================
# Gesture tracking
# ============================================================

class HandTracker:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "MediaPipe model not found:\n"
                f"{MODEL_PATH}\n\n"
                "Download hand_landmarker.task into the "
                "project directory."
            )

        base_options = python.BaseOptions(
            model_asset_path=MODEL_PATH
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )

        self.landmarker = (
            vision.HandLandmarker.create_from_options(
                options
            )
        )

        self.timestamp_ms = 0

        self.right_index = None
        self.left_wrist = None

        self.right_landmarks = None
        self.left_landmarks = None

        self.right_label = None
        self.left_label = None

    def close(self):
        self.landmarker.close()

    @staticmethod
    def distance(a, b):
        ax = float(a.x)
        ay = float(a.y)
        bx = float(b.x)
        by = float(b.y)

        return float(np.hypot(ax - bx, ay - by))

    def is_pinch(self, landmarks):
        if landmarks is None:
            return False

        thumb = landmarks[4]
        index = landmarks[8]

        # Relative to palm scale.
        wrist = landmarks[0]
        middle_mcp = landmarks[9]

        palm_size = self.distance(
            wrist,
            middle_mcp,
        )

        if palm_size < 1e-6:
            return False

        pinch_distance = self.distance(
            thumb,
            index,
        )

        return (
            pinch_distance / palm_size
            < PINCH_ON
        )

    def process(self, frame):
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        self.timestamp_ms += 33

        result = self.landmarker.detect_for_video(
            mp_image,
            self.timestamp_ms,
        )

        self.right_index = None
        self.left_wrist = None
        self.right_landmarks = None
        self.left_landmarks = None

        if not result.hand_landmarks:
            return result

        for idx, landmarks in enumerate(
            result.hand_landmarks
        ):

            handedness = (
                result.handedness[idx][0].category_name
            )

            if handedness.lower() == "right":
                self.right_landmarks = landmarks

                tip = landmarks[8]

                self.right_index = (
                    int(tip.x * frame.shape[1]),
                    int(tip.y * frame.shape[0]),
                )

                self.right_label = handedness

            elif handedness.lower() == "left":
                self.left_landmarks = landmarks

                wrist = landmarks[0]

                self.left_wrist = (
                    int(wrist.x * frame.shape[1]),
                    int(wrist.y * frame.shape[0]),
                )

                self.left_label = handedness

        return result


# ============================================================
# Gesture chess controller
# ============================================================

class GestureChessController:
    def __init__(self, board):
        self.board = board

        self.hover_square = None
        self.selected_square = None

        self.pinch_active = False
        self.pinch_started = False

        self.pinch_frames = 0
        self.release_frames = 0

        self.fingertip = None
        self.fingertip_smooth = None

        self.trail = deque(
            maxlen=TRAIL_LENGTH
        )

        self.status = "POINT AT A PIECE"

        self.last_move = None

        self.pending_promotion = None
        self.promotion_choice = chess.QUEEN

        self.last_action_time = 0.0
        self.computer_thinking = False

    def _choose_black_move(self):
        if self.board.turn != chess.BLACK:
            return None

        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return None

        best_move = legal_moves[0]
        best_score = -float("inf")

        for move in legal_moves:
            self.board.push(move)
            score = self._evaluate_position(1, False)
            self.board.pop()

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _evaluate_position(self, depth, is_black_turn):
        legal_moves = list(self.board.legal_moves)
        if depth == 0 or not legal_moves:
            return self._board_score()

        if is_black_turn:
            best = float("inf")
            for move in legal_moves:
                self.board.push(move)
                score = self._evaluate_position(depth - 1, False)
                self.board.pop()
                best = min(best, score)
            return best

        best = -float("inf")
        for move in legal_moves:
            self.board.push(move)
            score = self._evaluate_position(depth - 1, True)
            self.board.pop()
            best = max(best, score)
        return best

    def _board_score(self):
        score = 0
        for square, piece in self.board.piece_map().items():
            value = {
                chess.PAWN: 100,
                chess.KNIGHT: 320,
                chess.BISHOP: 330,
                chess.ROOK: 500,
                chess.QUEEN: 900,
                chess.KING: 20000,
            }.get(piece.piece_type, 0)

            if piece.color == chess.WHITE:
                score += value
            else:
                score -= value

        return score

    def choose_computer_move(self):
        if self.board.turn != chess.BLACK:
            return None
        return self._choose_black_move()

    def update_fingertip(self, point):
        if point is None:
            self.fingertip = None
            return

        self.fingertip = point

        if self.fingertip_smooth is None:
            self.fingertip_smooth = (
                float(point[0]),
                float(point[1]),
            )
        else:
            alpha = FINGER_EMA

            self.fingertip_smooth = (
                self.fingertip_smooth[0]
                * (1 - alpha)
                + point[0] * alpha,

                self.fingertip_smooth[1]
                * (1 - alpha)
                + point[1] * alpha,
            )

        self.trail.append(
            (
                int(self.fingertip_smooth[0]),
                int(self.fingertip_smooth[1]),
            )
        )

    def update_hover(self, board_flipped):
        if self.fingertip_smooth is None:
            self.hover_square = None
            return

        self.hover_square = screen_to_square(
            self.fingertip_smooth,
            board_flipped,
        )

    def pinch_update(
        self,
        is_pinch,
        board_flipped,
    ):
        if is_pinch:
            self.pinch_frames += 1
            self.release_frames = 0

            if (
                not self.pinch_active
                and self.pinch_frames
                >= PINCH_STABLE_FRAMES
            ):
                self.pinch_active = True
                self.pinch_started = True
                self.on_pinch_start()

        else:
            self.release_frames += 1
            self.pinch_frames = 0

            if (
                self.pinch_active
                and self.release_frames
                >= RELEASE_STABLE_FRAMES
            ):
                self.pinch_active = False
                self.pinch_started = False
                self.on_pinch_release(board_flipped)

    def on_pinch_start(self):
        if self.board.turn != chess.WHITE:
            self.status = "COMPUTER THINKING..."
            return

        if self.hover_square is None:
            self.status = "POINT AT A PIECE"
            return

        piece = self.board.piece_at(
            self.hover_square
        )

        if piece is None:
            self.status = "EMPTY SQUARE"
            return

        if piece.color != self.board.turn:
            self.status = "NOT YOUR TURN"
            return

        self.selected_square = self.hover_square
        self.status = (
            f"SELECTED {chess.square_name(self.selected_square)}"
        )

    def on_pinch_release(
        self,
        board_flipped,
    ):
        if self.selected_square is None:
            self.status = "POINT AT A PIECE"
            return

        target = self.hover_square

        if target is None:
            self.status = "MOVE CANCELLED"
            self.selected_square = None
            return

        self.attempt_move(
            self.selected_square,
            target,
        )

    def _play_computer_move(self):
        if self.board.turn != chess.BLACK or self.board.is_game_over():
            return False

        move = self.choose_computer_move()
        if move is None:
            self.status = "NO LEGAL MOVE"
            return False

        san = self.board.san(move)
        self.board.push(move)
        self.last_move = san

        if self.board.is_checkmate():
            self.status = "CHECKMATE"
        elif self.board.is_stalemate():
            self.status = "STALEMATE"
        elif self.board.is_check():
            self.status = f"COMPUTER CHECK: {san}"
        else:
            self.status = f"COMPUTER: {san}"

        return True

    def attempt_move(
        self,
        from_square,
        to_square,
    ):
        if self.board.turn != chess.WHITE:
            self.status = "COMPUTER THINKING..."
            return False

        piece = self.board.piece_at(
            from_square
        )

        if piece is None:
            self.status = "NO PIECE"
            self.selected_square = None
            return False

        promotion = None
        if (
            piece.piece_type == chess.PAWN
            and chess.square_rank(to_square)
            in (0, 7)
        ):
            promotion = self.promotion_choice

        move = chess.Move(
            from_square,
            to_square,
            promotion=promotion,
        )

        if move not in self.board.legal_moves:
            if (
                piece.piece_type == chess.PAWN
                and chess.square_rank(to_square)
                in (0, 7)
            ):
                move = chess.Move(
                    from_square,
                    to_square,
                    promotion=self.promotion_choice,
                )

            if move not in self.board.legal_moves:
                self.status = "ILLEGAL MOVE"
                self.selected_square = None
                return False

        san = self.board.san(move)
        self.board.push(move)
        self.last_move = san
        self.selected_square = None
        self.last_action_time = time.time()

        if self.board.is_checkmate():
            self.status = "CHECKMATE"
            return True

        if self.board.is_stalemate():
            self.status = "STALEMATE"
            return True

        if self.board.is_check():
            self.status = f"CHECK: {san}"
        else:
            self.status = f"MOVED {san}"

        if self.board.turn == chess.BLACK:
            self.status = "COMPUTER THINKING..."
            self._play_computer_move()

        return True

    def legal_targets(self):
        if self.selected_square is None:
            return set()

        return {
            move.to_square
            for move in self.board.legal_moves
            if move.from_square
            == self.selected_square
        }


# ============================================================
# Mouse fallback
# ============================================================

class MouseController:
    def __init__(self):
        self.pending_square = None
        self.enabled = True


# ============================================================
# Left-hand board orbit
# ============================================================

class OrbitController:
    def __init__(self):
        self.active = False
        self.last_point = None

    @staticmethod
    def is_open_palm(landmarks):
        if landmarks is None:
            return False

        wrist = landmarks[0]

        finger_tips = [
            landmarks[8],
            landmarks[12],
            landmarks[16],
            landmarks[20],
        ]

        finger_mcps = [
            landmarks[5],
            landmarks[9],
            landmarks[13],
            landmarks[17],
        ]

        extended = 0

        for tip, mcp in zip(
            finger_tips,
            finger_mcps,
        ):
            if tip.y < mcp.y:
                extended += 1

        return extended >= 3

    def update(self, landmarks):
        if landmarks is None:
            self.active = False
            self.last_point = None
            return

        wrist = landmarks[0]

        point = (
            float(wrist.x),
            float(wrist.y),
        )

        open_palm = self.is_open_palm(
            landmarks
        )

        if not open_palm:
            self.active = False
            self.last_point = None
            return

        if self.last_point is None:
            self.last_point = point
            self.active = True
            return

        dx = point[0] - self.last_point[0]
        dy = point[1] - self.last_point[1]

        global ROT_Y
        global ROT_Z

        ROT_Y += (
            dx * 100
            * ORBIT_SENSITIVITY_X
        )

        ROT_Z += (
            dy * 100
            * ORBIT_SENSITIVITY_Y
        )

        ROT_Y = max(
            -55,
            min(55, ROT_Y),
        )

        ROT_Z = max(
            -35,
            min(35, ROT_Z),
        )

        self.last_point = point
        self.active = True


# ============================================================
# Score / hints / hand map
# ============================================================

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}


def board_material_score(board):
    white_score = 0
    black_score = 0

    for piece in board.piece_map().values():
        value = PIECE_VALUES.get(piece.piece_type, 0)
        if piece.color == chess.WHITE:
            white_score += value
        else:
            black_score += value

    return white_score, black_score


def build_hint(board):
    if board.is_game_over():
        return "Hint: game is over."

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return "Hint: no legal moves remain."

    move = legal_moves[0]
    try:
        san = board.san(move)
    except ValueError:
        san = move.uci()

    return f"Hint: {move.uci()} ({san})"


def draw_hand_landmarks(frame, landmarks, color=(60, 200, 255), radius=3):
    if landmarks is None:
        return

    panel_w = 210
    panel_h = 150
    pad = 18
    x0 = frame.shape[1] - panel_w - pad
    y0 = frame.shape[0] - panel_h - pad

    cv2.rectangle(
        frame,
        (x0, y0),
        (x0 + panel_w, y0 + panel_h),
        (18, 18, 18),
        -1,
    )
    cv2.rectangle(
        frame,
        (x0, y0),
        (x0 + panel_w, y0 + panel_h),
        (120, 120, 120),
        1,
    )

    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)

    for connection in mp.solutions.hands.HAND_CONNECTIONS:
        p1 = landmarks[connection[0]]
        p2 = landmarks[connection[1]]

        x1 = int(p1.x * panel_w)
        y1 = int(p1.y * panel_h)
        x2 = int(p2.x * panel_w)
        y2 = int(p2.y * panel_h)

        cv2.line(panel, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    for landmark in landmarks:
        x = int(landmark.x * panel_w)
        y = int(landmark.y * panel_h)
        cv2.circle(panel, (x, y), radius, color, -1, cv2.LINE_AA)

    frame[y0:y0 + panel_h, x0:x0 + panel_w] = panel


def draw_hud(
    frame,
    board,
    controller,
    tracker,
    orbit,
):
    # Top-left status

    turn_text = (
        "WHITE TO MOVE"
        if board.turn == chess.WHITE
        else "BLACK TO MOVE"
    )

    if board.is_checkmate():
        turn_text = "CHECKMATE"

    elif board.is_stalemate():
        turn_text = "STALEMATE"

    elif board.is_check():
        turn_text += "  |  CHECK"

    cv2.rectangle(
        frame,
        (20, 18),
        (490, 185),
        (15, 15, 15),
        -1,
    )

    cv2.rectangle(
        frame,
        (20, 18),
        (490, 185),
        (100, 100, 100),
        1,
    )

    cv2.putText(
        frame,
        "GESTURE CHESS",
        (35, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        WHITE,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        turn_text,
        (35, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        GREEN if board.turn == chess.WHITE else YELLOW,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        controller.status,
        (35, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        WHITE,
        1,
        cv2.LINE_AA,
    )

    white_score, black_score = board_material_score(board)
    cv2.putText(
        frame,
        f"You: {white_score}   Computer: {black_score}",
        (35, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        GREEN if board.turn == chess.WHITE else YELLOW,
        1,
        cv2.LINE_AA,
    )

    hint = build_hint(board)
    cv2.putText(
        frame,
        hint,
        (35, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        GRAY,
        1,
        cv2.LINE_AA,
    )

    last_move = (
        controller.last_move
        if controller.last_move
        else "-"
    )

    cv2.putText(
        frame,
        f"Last move: {last_move}",
        (35, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        GRAY,
        1,
        cv2.LINE_AA,
    )

    # Gesture status

    right_status = "RIGHT: CURSOR"

    left_status = (
        "LEFT: OPTIONAL ROTATE"
        if orbit.active
        else "LEFT: IDLE"
    )

    cv2.putText(
        frame,
        right_status,
        (W_FRAME - 280, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        GREEN if controller.pinch_active else WHITE,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        left_status,
        (W_FRAME - 280, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        BLUE if orbit.active else GRAY,
        2,
        cv2.LINE_AA,
    )

    # Controls

    controls = [
        "Right index: cursor",
        "Pinch once: select",
        "Move + pinch again: place",
        "Left hand: optional rotate",
        "R reset   U undo   F flip",
        "ESC / Q quit",
    ]

    base_y = H_FRAME - 145

    cv2.rectangle(
        frame,
        (
            20,
            base_y - 15,
        ),
        (
            320,
            H_FRAME - 20,
        ),
        (15, 15, 15),
        -1,
    )

    for i, text in enumerate(controls):
        cv2.putText(
            frame,
            text,
            (
                35,
                base_y + i * 20,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            WHITE,
            1,
            cv2.LINE_AA,
        )


# ============================================================
# Highlights
# ============================================================

def draw_highlights(
    frame,
    controller,
    board_flipped,
):
    # Hover

    if controller.hover_square is not None:

        file_index = chess.square_file(
            controller.hover_square
        )

        rank_index = chess.square_rank(
            controller.hover_square
        )

        if board_flipped:
            file_index = 7 - file_index
            rank_index = 7 - rank_index

        corners = square_corners(
            file_index,
            rank_index,
            BOARD_Z + 0.015,
        )

        screen = project_points(
            [
                transform_point(p)
                for p in corners
            ]
        )

        overlay = frame.copy()

        highlight_fill = (120, 230, 255)
        draw_polygon(
            overlay,
            screen,
            highlight_fill,
            None,
        )

        cv2.addWeighted(
            overlay,
            0.35,
            frame,
            0.65,
            0,
            frame,
        )

        cv2.polylines(
            frame,
            [
                np.asarray(
                    screen,
                    dtype=np.int32,
                )
            ],
            True,
            (255, 255, 102),
            4,
            cv2.LINE_AA,
        )

    # Legal targets

    for square in controller.legal_targets():

        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)

        if board_flipped:
            file_index = 7 - file_index
            rank_index = 7 - rank_index

        center = world_to_screen(
            square_center(
                file_index,
                rank_index,
                BOARD_Z + 0.04,
            )
        )

        cv2.circle(
            frame,
            center,
            9,
            GREEN,
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            frame,
            center,
            9,
            BLACK,
            2,
            cv2.LINE_AA,
        )


# ============================================================
# Fingertip rendering
# ============================================================

def draw_gesture_overlay(
    frame,
    controller,
):
    if controller.fingertip_smooth is None:
        return

    for i in range(
        1,
        len(controller.trail),
    ):
        p1 = controller.trail[i - 1]
        p2 = controller.trail[i]

        thickness = max(
            1,
            i // 4,
        )

        cv2.line(
            frame,
            p1,
            p2,
            (120, 210, 255),
            thickness,
            cv2.LINE_AA,
        )

    point = (
        int(controller.fingertip_smooth[0]),
        int(controller.fingertip_smooth[1]),
    )

    cv2.circle(
        frame,
        point,
        13,
        YELLOW,
        2,
        cv2.LINE_AA,
    )

    if controller.pinch_active:

        cv2.circle(
            frame,
            point,
            8,
            GREEN,
            -1,
            cv2.LINE_AA,
        )


# ============================================================
# Mouse callback
# ============================================================

mouse_state = {
    "point": None,
    "clicked": False,
}


def mouse_callback(
    event,
    x,
    y,
    flags,
    param,
):
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_state["point"] = (
            x,
            y,
        )
        mouse_state["clicked"] = True


# ============================================================
# Application
# ============================================================

def create_board():
    return chess.Board()


def reset_controller(controller):
    controller.hover_square = None
    controller.selected_square = None
    controller.pinch_active = False
    controller.pinch_started = False
    controller.pinch_frames = 0
    controller.release_frames = 0
    controller.status = "POINT AT A PIECE"
    controller.last_move = None
    controller.pending_promotion = None


def undo_move(board, controller):
    if board.move_stack:
        board.pop()
        controller.selected_square = None
        controller.last_move = "UNDO"
        controller.status = "MOVE UNDONE"


def main():
    global ROT_X
    global ROT_Y
    global ROT_Z

    print("=" * 60)
    print("GESTURE CHESS")
    print("=" * 60)
    print("Python:", sys.version.split()[0])
    print("python-chess:", chess.__version__)
    print("Model:", MODEL_PATH)
    print()
    print("Controls:")
    print("  Right index       -> hover")
    print("  Pinch             -> select")
    print("  Pinch + move      -> drag")
    print("  Release           -> move")
    print("  Left open palm    -> orbit")
    print("  R                 -> reset")
    print("  U                 -> undo")
    print("  F                 -> flip board")
    print("  Q / ESC           -> quit")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(
            "\nERROR: hand_landmarker.task not found."
        )
        return 1

    board = create_board()

    controller = GestureChessController(
        board
    )

    tracker = HandTracker()
    orbit = OrbitController()

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not camera.isOpened():
        print(
            "ERROR: Could not open webcam."
        )

        tracker.close()
        return 1

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        W_FRAME,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        H_FRAME,
    )

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.setWindowProperty(
        WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )

    cv2.setMouseCallback(
        WINDOW_NAME,
        mouse_callback,
    )

    board_flipped = False

    try:

        while True:

            ok, frame = camera.read()

            if not ok:
                print(
                    "ERROR: Could not read webcam frame."
                )
                break

            # Mirror camera for natural interaction.
            frame = cv2.flip(
                frame,
                1,
            )

            frame = cv2.resize(
                frame,
                (W_FRAME, H_FRAME),
            )

            # ------------------------------------------------
            # Hand tracking
            # ------------------------------------------------

            tracker.process(frame)

            controller.update_fingertip(
                tracker.right_index
            )

            controller.update_hover(
                board_flipped
            )

            right_pinch = tracker.is_pinch(
                tracker.right_landmarks
            )

            controller.pinch_update(
                right_pinch,
                board_flipped,
            )

            # Left hand orbit
            orbit.update(
                tracker.left_landmarks
            )

            # ------------------------------------------------
            # Mouse fallback
            # ------------------------------------------------

            if mouse_state["clicked"]:

                mouse_point = mouse_state[
                    "point"
                ]

                mouse_square = (
                    screen_to_square(
                        mouse_point,
                        board_flipped,
                    )
                )

                if mouse_square is not None:

                    if (
                        controller.selected_square
                        is None
                    ):
                        piece = board.piece_at(
                            mouse_square
                        )

                        if (
                            piece is not None
                            and piece.color
                            == board.turn
                        ):
                            controller.selected_square = (
                                mouse_square
                            )

                            controller.status = (
                                f"SELECTED "
                                f"{chess.square_name(mouse_square)}"
                            )

                    else:

                        controller.attempt_move(
                            controller.selected_square,
                            mouse_square,
                        )

                mouse_state["clicked"] = False

            # ------------------------------------------------
            # Render board
            # ------------------------------------------------

            draw_board(
                frame,
                board_flipped,
            )

            draw_highlights(
                frame,
                controller,
                board_flipped,
            )

            draw_pieces(
                frame,
                board,
                controller.selected_square,
                board_flipped,
            )

            draw_hand_landmarks(
                frame,
                tracker.right_landmarks,
                (60, 200, 255),
                4,
            )
            draw_hand_landmarks(
                frame,
                tracker.left_landmarks,
                (130, 255, 130),
                4,
            )

            draw_gesture_overlay(
                frame,
                controller,
            )

            draw_hud(
                frame,
                board,
                controller,
                tracker,
                orbit,
            )

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            cv2.imshow(
                WINDOW_NAME,
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (
                27,
                ord("q"),
                ord("Q"),
            ):
                break

            elif key in (
                ord("r"),
                ord("R"),
            ):
                board = create_board()
                controller.board = board
                reset_controller(controller)

            elif key in (
                ord("u"),
                ord("U"),
            ):
                undo_move(
                    board,
                    controller,
                )

            elif key in (
                ord("f"),
                ord("F"),
            ):
                board_flipped = not board_flipped

            elif key == ord("1"):
                controller.promotion_choice = (
                    chess.QUEEN
                )

            elif key == ord("2"):
                controller.promotion_choice = (
                    chess.ROOK
                )

            elif key == ord("3"):
                controller.promotion_choice = (
                    chess.BISHOP
                )

            elif key == ord("4"):
                controller.promotion_choice = (
                    chess.KNIGHT
                )

    finally:

        camera.release()
        tracker.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )