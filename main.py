import sys
import cv2


from core.camera     import CameraCapture
from core.pose_engine import PoseEngine
from core.renderer   import draw_fps, draw_skeleton, draw_text, RED
from games.menu      import MainMenu
from games.boxing    import BoxingGame
from games.pose_challenge import PoseChallenge
from utils.landmarks import POSE_CONNECTIONS

FRAME_W    = 640
FRAME_H    = 480
MODEL_PATH = "assets/models/pose_landmarker_lite.task"
FPS_SKIP_THRESHOLD = 25.0   # below this FPS → process every other frame


def main() -> None:
    cam = CameraCapture(cam_id=0, width=FRAME_W, height=FRAME_H)
    if not cam.cap.isOpened():
        print("ERROR: No se pudo abrir la camara.", file=sys.stderr)
        return

    engine = PoseEngine(model_path=MODEL_PATH)

    games = {
        "menu":          MainMenu(FRAME_W, FRAME_H),
        "boxing":        BoxingGame(FRAME_W, FRAME_H),
        "pose_challenge":PoseChallenge(FRAME_W, FRAME_H),
    }
    state     = "menu"
    landmarks = None
    frame_n   = 0
    skip_ai   = False

    cv2.namedWindow("Consola Multijuegos", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Consola Multijuegos", FRAME_W, FRAME_H)

    while True:
        frame = cam.read()
        if frame is None:
            draw_text(
                __import__("numpy").zeros((FRAME_H, FRAME_W, 3), dtype=__import__("numpy").uint8),
                "ERROR: camara desconectada", (80, FRAME_H // 2), color=RED,
            )
            break

        frame_n += 1

        # Adaptive AI: skip every other frame when FPS < threshold
        if not skip_ai or frame_n % 2 == 0:
            landmarks = engine.process(frame)

        game = games[state]
        game.update(frame, landmarks, FRAME_W, FRAME_H)

        if landmarks:
            draw_skeleton(frame, landmarks, POSE_CONNECTIONS, FRAME_W, FRAME_H)

        rendered = game.render(frame)
        if isinstance(rendered, type(frame)) and rendered is not None:
            frame = rendered

        draw_fps(frame, cam.fps)

        # State transition
        nxt = game.next_state
        if nxt == "exit":
            break
        if nxt and nxt in games:
            game.reset()
            state = nxt
            games[state].reset()

        # Re-evaluate adaptive skip every 30 frames
        if frame_n % 30 == 0 and cam.fps > 0:
            skip_ai = cam.fps < FPS_SKIP_THRESHOLD

        cv2.imshow("Consola Multijuegos", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:          # ESC → quit
            break
        elif key == ord("m"):  # M   → force menu
            games[state].reset()
            state = "menu"
            games["menu"].reset()

    cam.release()
    engine.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
