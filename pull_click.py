#pip install pypiwin32

import os
import sys
import cv2
import numpy as np
import pyautogui
import ctypes
import time
import win32gui
import win32con
import subprocess

def is_admin():
    """
    Check if the current process has administrative privileges.
    """
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False
def run_as_admin():
    """
    Restart the script with administrative privileges if not already admin.
    """
    if not is_admin():
        try:
            script = os.path.abspath(sys.argv[0])
            params = ' '.join([f'"{arg}"' for arg in sys.argv[1:]])
            # Re-launch the process with administrator privileges
            ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, f'"{script}" {params}', None, 1)
            sys.exit(0)
        except Exception as e:
            print(f"Failed to elevate privileges: {e}")
            sys.exit(1)

def enum_windows_callback(hwnd, target_title_fragment):
    """
    Callback function for win32gui.EnumWindows to bring a window with a specific title fragment to the front.
    """
    if win32gui.IsWindowVisible(hwnd):
        window_text = win32gui.GetWindowText(hwnd)
        if target_title_fragment in window_text:
            try:
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)  # Restore if minimized
                win32gui.SetForegroundWindow(hwnd)
                print(f"Window '{window_text}' brought to the front.")
                return False  # Stop enumeration once the target window is found
            except Exception as e:
                print(f"Error bringing window to front: {e}")

    return True

def bring_window_to_front(target_title_fragment):
    """
    Brings the first window containing the target title fragment to the foreground.
    """
    win32gui.EnumWindows(enum_windows_callback, target_title_fragment)

def click_with_ctypes(x, y):
    """
    Click at a specific location using ctypes-based low-level Windows API calls.
    """
    ctypes.windll.user32.SetCursorPos(x, y)
    time.sleep(0.1)  # Small delay to ensure cursor position is recognized
    ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)  # Mouse left button down
    ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)  # Mouse left button up

def find_button_and_click(template_path, interval_minutes, duration_minutes, target_title_fragment):
    """
    Finds an image on the screen, brings the application window to the front, clicks using ctypes, 
    and presses Enter after 1 second delay.
    """
    interval_seconds = interval_minutes * 60
    total_seconds = duration_minutes * 60
    start_time = time.time()

    print("Starting the button image recognition and clicking process...")
    while time.time() - start_time < total_seconds:
        # Bring window to the front
        bring_window_to_front(target_title_fragment)

        # Capture screenshot and search for the template
        screenshot = np.array(pyautogui.screenshot())
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        template = cv2.imread(template_path, 0)

        result = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val >= 0.8:  # Adjust the threshold value as needed.
            # Locate the center of the matched region
            template_height, template_width = template.shape
            center_x, center_y = max_loc[0] + template_width // 2, max_loc[1] + template_height // 2

            click_with_ctypes(center_x, center_y)
            print(f"Clicked the button at ({center_x}, {center_y})")
            
            # Wait for 1 second
            time.sleep(1)
            
            # Press Enter
            pyautogui.press("enter")
            print("Pressed Enter key")
            time.sleep(10)
            pyautogui.press("enter")
        else:
            print("Button not found on screen.")

        time.sleep(interval_seconds)

# if __name__ == "__main__":
#     # Check and request for admin privileges
#     run_as_admin()

#     # Path to your image file of the button
#     template_image_path = "pull.png"

#     # Set intervals and durations for clicks
#     interval_minutes = 5  # Click every 5 minutes
#     duration_minutes = 60  # Run script for 60 minutes

#     # Specify part of the window title to bring it to the front
#     partial_title = "주식회사 코린토"

#     find_button_and_click(template_image_path, interval_minutes, duration_minutes, partial_title)