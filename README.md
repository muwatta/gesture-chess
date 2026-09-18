# Gesture Chess

A virtual chess game you control with your hands.

**Gesture Chess** combines webcam-based hand tracking, MediaPipe, OpenCV, and `python-chess` to create an interactive chess experience where you play against the computer without needing a physical chessboard.

Your **right index finger acts as a virtual cursor**. Point at a chess square, pinch to select a piece, move your finger to a legal destination, and release to make the move.

You play as **White** and the computer plays as **Black**.

## Features

* Webcam-based hand tracking with MediaPipe
* Gesture-controlled chess interaction
* Visible fingertip cursor and gesture feedback
* Point to hover over chess squares
* Pinch to select a piece
* Drag a selected piece using your finger
* Release to attempt a move
* Legal chess moves enforced by `python-chess`
* Captures, check, checkmate, castling, en passant, and promotion
* Human vs Computer gameplay
* Computer automatically responds after your move
* Visual highlighting of selected pieces and legal destinations
* Virtual chessboard rendered over the webcam feed
* Keyboard controls as a fallback
* Board reset and undo support
* Board orientation can be flipped

## Controls

| Input      | Gesture / Key                | Action                  |
| ---------- | ---------------------------- | ----------------------- |
| Right hand | Index finger                 | Move the virtual cursor |
| Right hand | Move index finger over board | Hover a chess square    |
| Right hand | Thumb + index pinch          | Select a piece          |
| Right hand | Move while selected          | Drag the selected piece |
| Right hand | Release pinch                | Attempt the move        |
| Keyboard   | `R`                          | Reset the game          |
| Keyboard   | `U`                          | Undo the last move      |
| Keyboard   | `F`                          | Flip the board          |
| Keyboard   | `Q` / `ESC`                  | Quit                    |

### Chess Flow

The intended interaction is:

```text
Point
  ↓
Hover a piece
  ↓
Pinch
  ↓
Piece selected
  ↓
Move your finger
  ↓
Destination highlighted
  ↓
Release
  ↓
Legal move
  ↓
Computer responds
```

You are always **White**, so you make the first move.

The computer controls **Black** and automatically makes its move after your legal move.

## Requirements

* Python 3.9+
* A working webcam
* MediaPipe
* OpenCV
* NumPy
* `python-chess`
* A MediaPipe Hand Landmarker model

The project includes `hand_landmarker.task` locally. If it is not present, download it from the official MediaPipe model repository.

## Setup

Clone the repository:

```bash
git clone https://github.com/muwatta/gesture-chess.git
cd gesture-chess
```

Create and activate a virtual environment.

### Git Bash / Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows Command Prompt

```bat
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

If the hand-tracking model is not already present:

```bash
curl -L -o hand_landmarker.task \
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Run

Start the game with:

```bash
python gesture_chess.py
```

The webcam window will open and the virtual chessboard will appear over the camera feed.

Make sure your right hand is visible to the webcam and keep your index finger clearly separated from your other fingers when pointing.

## How to Play

### 1. Point

Raise your right hand and extend your index finger.

Move your fingertip over the chessboard.

A visual marker follows your fingertip and identifies the square you are pointing at.

### 2. Select

Move the marker over one of your White pieces.

Pinch your **thumb and index finger together**.

The piece becomes selected and its legal destination squares are highlighted.

### 3. Drag

Keep the pinch active and move your index finger toward the destination square.

You do not need to physically touch the screen or chessboard.

### 4. Release

Release the pinch.

If the move is legal, the piece moves.

If the move is illegal, the board remains unchanged.

### 5. Computer Turn

After your legal move, the computer automatically plays Black.

The game then returns to your turn.

## Chess Rules

Chess legality is handled by [`python-chess`](https://python-chess.readthedocs.io/).

This means the application does not manually implement individual chess rules.

The game supports standard chess mechanics including:

* Legal move validation
* Piece movement
* Captures
* Check
* Checkmate
* Stalemate
* Castling
* En passant
* Pawn promotion
* Turn management
* Game-over detection

## Keyboard Fallback

Gesture interaction is the primary interface, but keyboard controls are available for convenience and testing.

```text
R      Reset game
U      Undo move
F      Flip board
Q      Quit
ESC    Quit
```

## Project Structure

```text
gesture-chess/
├── gesture_chess.py
├── hand_landmarker.task
├── requirements.txt
├── README.md
└── LICENSE
```

## Testing

Run the test suite with:

```bash
pip install pytest
pytest -v
```

The tests cover the chess game state and application logic without requiring a webcam.

The camera application is started only when `gesture_chess.py` is executed directly, so importing the module does not automatically open the webcam.

## Technology Stack

**Python** — Core application language

**OpenCV** — Webcam capture, rendering, drawing, and visual interface

**MediaPipe** — Real-time hand landmark detection and gesture tracking

**NumPy** — Numerical operations and geometry

**python-chess** — Chess board state and legal move validation

## Project Architecture

```text
Webcam
   ↓
MediaPipe Hand Tracking
   ↓
Gesture Detection
   ↓
Gesture Chess Controller
   ↓
Chess Board State
   ↓
Computer Move
   ↓
OpenCV Renderer
   ↓
Webcam + Virtual Chessboard
```

## Known Limitations

Gesture recognition depends on camera quality, lighting, hand position, and visibility.

The application works best when the right hand is clearly visible and the index finger and thumb can be distinguished by the camera.

Fast movements, poor lighting, significant occlusion, or placing the hand too close to the camera may reduce tracking accuracy.

The computer opponent is software-based and does not require an internet connection.

## License

MIT — see [LICENSE](LICENSE).
