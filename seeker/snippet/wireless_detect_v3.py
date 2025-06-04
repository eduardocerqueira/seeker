#date: 2025-06-04T16:49:12Z
#url: https://api.github.com/gists/37de9f8cd6c433422d8a96ac6ba653e5
#owner: https://api.github.com/users/EncodeTheCode

import win32gui
import win32con
import win32api
import ctypes
from ctypes import wintypes
import subprocess

# Define ListView constants
LVS_REPORT = 0x0001
LVS_SINGLESEL = 0x0004
LVCF_FMT = 0x0001
LVCF_WIDTH = 0x0002
LVCF_TEXT = 0x0004
LVCFMT_LEFT = 0x0000
LVM_INSERTCOLUMN = 0x101B
LVM_DELETEALLITEMS = 0x1009
LVM_INSERTITEM = 0x1007
LVM_SETITEMTEXT = 0x1026
LVIF_TEXT = 0x0001
WC_LISTVIEW = "SysListView32"
WS_EX_CLIENTEDGE = 0x0200

listview_hwnd = None

class LVCOLUMN(ctypes.Structure):
    _fields_ = [
        ('mask', wintypes.UINT),
        ('fmt', wintypes.INT),
        ('cx', wintypes.INT),
        ('pszText', wintypes.LPWSTR),
        ('cchTextMax', wintypes.INT),
        ('iSubItem', wintypes.INT),
    ]

class LVITEM(ctypes.Structure):
    _fields_ = [
        ('mask', wintypes.UINT),
        ('iItem', wintypes.INT),
        ('iSubItem', wintypes.INT),
        ('state', wintypes.UINT),
        ('stateMask', wintypes.UINT),
        ('pszText', wintypes.LPWSTR),
        ('cchTextMax', wintypes.INT),
        ('iImage', wintypes.INT),
        ('lParam', wintypes.LPARAM),
    ]

SendMessage = ctypes.windll.user32.SendMessageW

def get_wifi_list():
    try:
        output = subprocess.check_output("netsh wlan show networks mode=bssid", shell=True, encoding='utf-8')
    except subprocess.CalledProcessError:
        return []

    networks = []
    current = {}

    for line in output.splitlines():
        line = line.strip()
        if line.startswith("SSID ") and " : " in line:
            if current:
                networks.append(current)
            current = {'SSID': line.split(" : ")[-1]}
        elif line.startswith("Signal"):
            current['Signal'] = line.split(" : ")[-1]
        elif line.startswith("Authentication"):
            current['Authentication'] = line.split(" : ")[-1]

    if current:
        networks.append(current)

    return networks

def populate_listview(hwnd):
    SendMessage(hwnd, LVM_DELETEALLITEMS, 0, 0)
    networks = get_wifi_list()

    for i, net in enumerate(networks):
        item = LVITEM()
        item.mask = LVIF_TEXT
        item.iItem = i
        item.iSubItem = 0
        item.pszText = ctypes.c_wchar_p(net['SSID'])
        item.cchTextMax = len(net['SSID'])
        SendMessage(hwnd, LVM_INSERTITEM, 0, ctypes.byref(item))

        for j, key in enumerate(['Signal', 'Authentication'], start=1):
            subitem = LVITEM()
            subitem.iItem = i
            subitem.iSubItem = j
            text = net.get(key, "")
            subitem.mask = LVIF_TEXT
            subitem.pszText = ctypes.c_wchar_p(text)
            subitem.cchTextMax = len(text)
            SendMessage(hwnd, LVM_SETITEMTEXT, i, ctypes.byref(subitem))

def on_command(hwnd, msg, wparam, lparam):
    if wparam == 1001:
        populate_listview(listview_hwnd)
    return 0

def wndproc(hwnd, msg, wparam, lparam):
    if msg == win32con.WM_DESTROY:
        win32gui.PostQuitMessage(0)
    elif msg == win32con.WM_COMMAND:
        return on_command(hwnd, msg, wparam, lparam)
    return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

def create_listview(parent_hwnd):
    global listview_hwnd

    listview_hwnd = win32gui.CreateWindowEx(
        WS_EX_CLIENTEDGE,
        WC_LISTVIEW,
        None,
        win32con.WS_CHILD | win32con.WS_VISIBLE | LVS_REPORT | LVS_SINGLESEL,
        10, 10, 560, 300,
        parent_hwnd,
        1000,
        None,
        None,
    )

    for i, title in enumerate(["SSID", "Signal", "Authentication"]):
        col = LVCOLUMN()
        col.mask = LVCF_FMT | LVCF_WIDTH | LVCF_TEXT
        col.fmt = LVCFMT_LEFT
        col.cx = 180
        col.pszText = ctypes.c_wchar_p(title)
        col.cchTextMax = len(title)
        col.iSubItem = i
        SendMessage(listview_hwnd, LVM_INSERTCOLUMN, i, ctypes.byref(col))

def main():
    hInstance = win32api.GetModuleHandle()
    className = "WiFiScannerClass"

    wndClass = win32gui.WNDCLASS()
    wndClass.lpfnWndProc = wndproc
    wndClass.hInstance = hInstance
    wndClass.lpszClassName = className
    wndClass.hCursor = win32gui.LoadCursor(None, win32con.IDC_ARROW)
    wndClass.hbrBackground = win32con.COLOR_WINDOW + 1

    atom = win32gui.RegisterClass(wndClass)

    hwnd = win32gui.CreateWindowEx(
        0,
        atom,
        "Wi-Fi Access Point Scanner",
        win32con.WS_OVERLAPPEDWINDOW,
        100, 100, 600, 400,
        0, 0,
        hInstance,
        None,
    )

    create_listview(hwnd)

    win32gui.CreateWindow(
        "Button",
        "Refresh",
        win32con.WS_CHILD | win32con.WS_VISIBLE | win32con.BS_DEFPUSHBUTTON,
        10, 320, 100, 30,
        hwnd,
        1001,
        hInstance,
        None
    )

    populate_listview(listview_hwnd)

    win32gui.ShowWindow(hwnd, win32con.SW_SHOWNORMAL)
    win32gui.UpdateWindow(hwnd)

    msg = wintypes.MSG()
    while ctypes.windll.user32.GetMessageW(ctypes.byref(msg), 0, 0, 0) != 0:
        ctypes.windll.user32.TranslateMessage(ctypes.byref(msg))
        ctypes.windll.user32.DispatchMessageW(ctypes.byref(msg))

if __name__ == "__main__":
    main()
