#date: 2025-06-04T16:48:49Z
#url: https://api.github.com/gists/fd6fde762f622fa6df90af862bc35370
#owner: https://api.github.com/users/EncodeTheCode

import win32gui
import win32con
import win32api
import ctypes
from ctypes import wintypes
import subprocess
import re

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
    """Return list of dicts with keys: SSID, BSSID, Signal, Authentication"""
    try:
        output = subprocess.check_output("netsh wlan show networks mode=bssid", shell=True, encoding='utf-8')
    except subprocess.CalledProcessError:
        return []

    networks = []
    ssid = None
    authentication = None

    # Regexes for parsing
    ssid_re = re.compile(r"^SSID\s+\d+\s+:\s+(.*)$")
    auth_re = re.compile(r"^Authentication\s+:\s+(.*)$")
    bssid_re = re.compile(r"^BSSID\s+\d+\s+:\s+(.*)$")
    signal_re = re.compile(r"^Signal\s+:\s+(.*)$")

    current_bssid = None
    current_signal = None

    for line in output.splitlines():
        line = line.strip()
        m_ssid = ssid_re.match(line)
        if m_ssid:
            # New SSID block
            ssid = m_ssid.group(1)
            authentication = None
            continue

        m_auth = auth_re.match(line)
        if m_auth:
            authentication = m_auth.group(1)
            continue

        m_bssid = bssid_re.match(line)
        if m_bssid:
            current_bssid = m_bssid.group(1)
            current_signal = None
            continue

        m_signal = signal_re.match(line)
        if m_signal and current_bssid is not None:
            current_signal = m_signal.group(1)
            # Append current entry
            networks.append({
                'SSID': ssid or "",
                'BSSID': current_bssid,
                'Signal': current_signal or "",
                'Authentication': authentication or "",
            })
            current_bssid = None
            current_signal = None

    return networks

def populate_listview(hwnd):
    SendMessage(hwnd, LVM_DELETEALLITEMS, 0, 0)
    networks = get_wifi_list()

    for i, net in enumerate(networks):
        # Insert main item: SSID
        item = LVITEM()
        item.mask = LVIF_TEXT
        item.iItem = i
        item.iSubItem = 0
        item.pszText = ctypes.c_wchar_p(net['SSID'])
        SendMessage(hwnd, LVM_INSERTITEM, 0, ctypes.byref(item))

        # BSSID subitem
        bssid_subitem = LVITEM()
        bssid_subitem.iItem = i
        bssid_subitem.iSubItem = 1
        bssid_subitem.mask = LVIF_TEXT
        bssid_subitem.pszText = ctypes.c_wchar_p(net['BSSID'])
        SendMessage(hwnd, LVM_SETITEMTEXT, i, ctypes.byref(bssid_subitem))

        # Signal subitem
        signal_subitem = LVITEM()
        signal_subitem.iItem = i
        signal_subitem.iSubItem = 2
        signal_subitem.mask = LVIF_TEXT
        signal_subitem.pszText = ctypes.c_wchar_p(net['Signal'])
        SendMessage(hwnd, LVM_SETITEMTEXT, i, ctypes.byref(signal_subitem))

        # Authentication subitem
        auth_subitem = LVITEM()
        auth_subitem.iItem = i
        auth_subitem.iSubItem = 3
        auth_subitem.mask = LVIF_TEXT
        auth_subitem.pszText = ctypes.c_wchar_p(net['Authentication'])
        SendMessage(hwnd, LVM_SETITEMTEXT, i, ctypes.byref(auth_subitem))

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
        10, 10, 580, 320,
        parent_hwnd,
        1000,
        None,
        None,
    )

    # Create columns: SSID, BSSID, Signal, Authentication
    columns = [
        ("SSID", 150),
        ("BSSID", 180),
        ("Signal", 80),
        ("Authentication", 150),
    ]

    for i, (title, width) in enumerate(columns):
        col = LVCOLUMN()
        col.mask = LVCF_FMT | LVCF_WIDTH | LVCF_TEXT
        col.fmt = LVCFMT_LEFT
        col.cx = width
        col.pszText = ctypes.c_wchar_p(title)
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
        100, 100, 620, 400,
        0, 0,
        hInstance,
        None,
    )

    create_listview(hwnd)

    win32gui.CreateWindow(
        "Button",
        "Refresh",
        win32con.WS_CHILD | win32con.WS_VISIBLE | win32con.BS_DEFPUSHBUTTON,
        10, 340, 100, 30,
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
