#date: 2021-09-01T17:15:44Z
#url: https://api.github.com/gists/db95208749cf0edd576bf5041d50cd9e
#owner: https://api.github.com/users/ThomasSelvig

import pygame as pg

import mouse, keyboard
import win32gui, win32con, win32api
from PIL import ImageGrab


class WinApiClient:
	def __init__(self, hwnd=None):
		self.hwnd = hwnd or win32gui.GetActiveWindow()
		self.default_style = win32gui.GetWindowLong(self.hwnd, win32con.GWL_EXSTYLE)
		self.COLOR_KEY = 121, 121, 77

		self.set_layered_mode()
		self.set_transparency()
		self.set_always_toplevel()

	def set_always_toplevel(self):
		""" make window always appear on top of other windows """
		if self.hwnd is None:
			return False

		old_win_pos = win32gui.GetWindowRect(self.hwnd)
		win32gui.SetWindowPos(
			self.hwnd,
			-1,
			old_win_pos[0],
			old_win_pos[1],
			0,
			0,
			0x0001
		)
		return True

	def set_layered_mode(self):
		if self.hwnd is None:
			return False

		# make window transparent click-through
		win32gui.SetWindowLong(
			self.hwnd, 
			win32con.GWL_EXSTYLE,
			# enable LAYERED and TRANSPARENT bits
			self.default_style | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_LAYERED
		)
		return True

	def set_transparency(self, opacity=1):
		# this method requires layered mode (WS_EX_LAYERED)
		if self.hwnd is None:
			return False

		win32gui.SetLayeredWindowAttributes(
			self.hwnd,
			win32api.RGB(*self.COLOR_KEY),
			int(opacity * 255),
			win32con.LWA_ALPHA | win32con.LWA_COLORKEY
		)
		return True


def screenshot_window(window_title):
	hwnd = win32gui.FindWindow(None, window_title)
	return ImageGrab.grab(win32gui.GetWindowRect(hwnd))


pg.init()
size = 1920, 1080
screen = pg.display.set_mode(size, pg.NOFRAME)
winapi = WinApiClient()


while True:
	if keyboard.is_pressed("q"):
		exit()

	screen.fill(winapi.COLOR_KEY)
	
	mpos = mouse.get_position()
	circle = pg.draw.circle(screen, (255, 0, 0), mpos, 25)

	pg.display.flip()
