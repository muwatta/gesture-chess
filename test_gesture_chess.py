import chess

from gesture_chess import GestureChessController, board_material_score, build_hint


def test_material_score_reflects_player_and_computer_materials():
    board = chess.Board("8/8/8/8/8/8/8/K7 w - - 0 1")

    white_score, black_score = board_material_score(board)

    assert white_score > black_score
    assert black_score == 0


def test_build_hint_returns_a_real_white_move_hint():
    board = chess.Board()

    hint = build_hint(board)

    assert isinstance(hint, str)
    assert "Hint" in hint


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

    expected_board = chess.Board()
    expected_board.push(chess.Move.from_uci("e2e4"))

    assert ok is True
    assert board.turn == chess.WHITE
    assert controller.last_move is not None
    assert controller.last_move in {
        expected_board.san(move) for move in expected_board.legal_moves
    }
