from pynput import keyboard
import time
from threading import Thread, Event

class KeyboardListener:
    def __init__(self):
        self.pressed_keys = set()
        self.stop_event = Event()
        self.a = 1

    def start_keyboard_listening_thread(self):
        # Start listening for keyboard events
        self.keyboard_listen_thread = Thread(target=self.keyboard_listening, args=(self.stop_event,), daemon=True)
        self.keyboard_listen_thread.start()

    def on_press(self, key):
        try:
            # Add key to the set of pressed keys
            self.pressed_keys.add(key.char if hasattr(key, 'char') else key.name)
            print(f"Keys currently pressed: {self.pressed_keys}")
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            # Remove key from the set of pressed keys
            self.pressed_keys.discard(key.char if hasattr(key, 'char') else key.name)
            # print(f"Keys currently pressed: {self.pressed_keys}")
        except AttributeError:
            pass
        
    def keyboard_listening(self, stop_event):
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            while not stop_event.is_set():
                time.sleep(0.01)
            # listener.join()

    def stop(self):
        self.stop_event.set()
        self.keyboard_listen_thread.join()
        self.pressed_keys = set()
        print("KeyboardListener has stopped.")

   