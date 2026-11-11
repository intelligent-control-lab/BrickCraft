import sys
import select
import threading
from typing import Any


class KeyboardStopper:
    def __init__(self, controller: Any):
        self.controller = controller
        self.stop_event = threading.Event()
        self.thread = None
        self.triggered = False

    def start(self):
        self.stop_event.clear()
        self.triggered = False
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.1)

    def _monitor_loop(self):
        print(f"[KeyboardStopper] Press 'q' + Enter to STOP controller...")

        while not self.stop_event.is_set():
            # select.select(rlist, wlist, xlist, timeout)
            # check input every 0.1 second
            i, o, e = select.select([sys.stdin], [], [], 0.1)

            if i:
                line = sys.stdin.readline().strip()
                if line.lower() == 'q':
                    print("\n [KeyboardStopper] 'q' detected! Stopping Controller...")
                    self.triggered = True
                    self.controller.stop()
                    break
                else:
                    print(f"[KeyboardStopper] Unrecognized input: {line}. Press 'q' + Enter to STOP controller.")
