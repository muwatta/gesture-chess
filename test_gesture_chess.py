import chess

from gesture_chess import GestureChessController


def test_computer_move_is_legal_for_black_turn():
    board = chess.Board("k7/8/8/8/8/8/8/K7 b - - 0 1")
    controller = GestureChessController(board)

    move = controller.choose_computer_move()

    assert move in board.legal_moves
    assert move.from_square == chess.A8
    assert board.turn == chess.BLACK


def test_human_white_move_advances_turn_and_automatically_resolves_black_response():
    board = chess.Board()
    controller = GestureChessController(board)

    ok = controller.attempt_move(chess.E2, chess.E4)

    assert ok is True
    assert board.turn == chess.WHITE
    assert controller.last_move in {"e4", "Nf3", "c4", "f3", "g3", "c3", "b4"}
