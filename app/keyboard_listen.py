from pynput import keyboard

def on_press(key):
    # global variables to be changed by chars
    # global foo
    if key == keyboard.Key.space:
        print("SPACE")
    if hasattr(key, 'char'):
        print("key", key.char)
        # if key.char == 'a':
def on_release(key):
    # if key == keyboard.Key.space:
    print('{0} released'.format(key))
 
def keyboard_listen():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
