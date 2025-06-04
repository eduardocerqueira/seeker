#date: 2025-06-04T16:47:59Z
#url: https://api.github.com/gists/be09d20127f6ded8a3d042af81b1bc31
#owner: https://api.github.com/users/EncodeTheCode

import ctypes
from ctypes import wintypes
import sys

# Constants from wlanapi.h
ERROR_SUCCESS = 0
MAX_PHY_TYPE_NUMBER = 8

# WLAN API DLL
wlanapi = ctypes.WinDLL('wlanapi.dll')

# HANDLE type for Windows handles
HANDLE = wintypes.HANDLE

# GUID structure (for interface GUIDs)
class GUID(ctypes.Structure):
    _fields_ = [
        ('Data1', wintypes.DWORD),
        ('Data2', wintypes.WORD),
        ('Data3', wintypes.WORD),
        ('Data4', wintypes.BYTE * 8)
    ]

# WLAN_INTERFACE_INFO structure
class WLAN_INTERFACE_INFO(ctypes.Structure):
    _fields_ = [
        ("InterfaceGuid", GUID),
        ("strInterfaceDescription", wintypes.WCHAR * 256),
        ("isState", wintypes.DWORD)
    ]

# WLAN_INTERFACE_INFO_LIST structure
class WLAN_INTERFACE_INFO_LIST(ctypes.Structure):
    _fields_ = [
        ("NumberOfItems", wintypes.DWORD),
        ("Index", wintypes.DWORD),
        # Actually this is an array of WLAN_INTERFACE_INFO structures, so we will handle this manually
        # We will treat this as a pointer to array of WLAN_INTERFACE_INFO following the two DWORDs
    ]

# WLAN_AVAILABLE_NETWORK structure
class WLAN_AVAILABLE_NETWORK(ctypes.Structure):
    _fields_ = [
        ("strProfileName", wintypes.WCHAR * 256),
        ("dot11Ssid", ctypes.c_byte * 32),  # We'll parse SSID manually below
        ("dot11BssType", wintypes.DWORD),
        ("uNumberOfBssids", wintypes.DWORD),
        ("bNetworkConnectable", wintypes.BOOL),
        ("wlanNotConnectableReason", wintypes.DWORD),
        ("uNumberOfPhyTypes", wintypes.DWORD),
        ("dot11PhyTypes", wintypes.DWORD * MAX_PHY_TYPE_NUMBER),
        ("bMorePhyTypes", wintypes.BOOL),
        ("wlanSignalQuality", wintypes.DWORD),
        ("bSecurityEnabled", wintypes.BOOL),
        ("dot11DefaultAuthAlgorithm", wintypes.DWORD),
        ("dot11DefaultCipherAlgorithm", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("dwReserved", wintypes.DWORD),
    ]

# WLAN_AVAILABLE_NETWORK_LIST structure (contains array of WLAN_AVAILABLE_NETWORK)
class WLAN_AVAILABLE_NETWORK_LIST(ctypes.Structure):
    _fields_ = [
        ("NumberOfItems", wintypes.DWORD),
        ("Index", wintypes.DWORD),
        # Followed by WLAN_AVAILABLE_NETWORK[NumberOfItems]
    ]

# dot11Ssid structure inside WLAN_AVAILABLE_NETWORK
class DOT11_SSID(ctypes.Structure):
    _fields_ = [
        ("uSSIDLength", wintypes.DWORD),
        ("ucSSID", ctypes.c_ubyte * 32)
    ]

def get_interface_list(client_handle):
    # typedef DWORD WINAPI WlanEnumInterfaces(
    #   HANDLE hClientHandle,
    #   PVOID pReserved,
    #   PWLAN_INTERFACE_INFO_LIST *ppInterfaceList
    # );
    ppInterfaceList = ctypes.POINTER(ctypes.c_void_p)()
    ret = wlanapi.WlanEnumInterfaces(client_handle, None, ctypes.byref(ppInterfaceList))
    if ret != ERROR_SUCCESS:
        print(f"WlanEnumInterfaces failed with error: {ret}")
        return None

    # ppInterfaceList is pointer to WLAN_INTERFACE_INFO_LIST
    iface_list_ptr = ctypes.cast(ppInterfaceList, ctypes.POINTER(WLAN_INTERFACE_INFO_LIST))

    # Number of interfaces
    num_interfaces = iface_list_ptr.contents.NumberOfItems

    # The interfaces start immediately after NumberOfItems and Index (DWORD each, 8 bytes total)
    base_ptr = ctypes.addressof(ppInterfaceList.contents) + ctypes.sizeof(WLAN_INTERFACE_INFO_LIST)

    interfaces = []
    for i in range(num_interfaces):
        # Calculate pointer to each WLAN_INTERFACE_INFO
        iface_ptr = ctypes.cast(base_ptr + i * ctypes.sizeof(WLAN_INTERFACE_INFO), ctypes.POINTER(WLAN_INTERFACE_INFO))
        interfaces.append(iface_ptr.contents)

    # Free memory allocated by WLAN API
    wlanapi.WlanFreeMemory(ppInterfaceList)

    return interfaces

def get_available_networks(client_handle, interface_guid):
    ppAvailableNetworkList = ctypes.POINTER(ctypes.c_void_p)()
    ret = wlanapi.WlanGetAvailableNetworkList(
        client_handle,
        ctypes.byref(interface_guid),
        0,  # flags
        None,  # reserved
        ctypes.byref(ppAvailableNetworkList)
    )
    if ret != ERROR_SUCCESS:
        print(f"WlanGetAvailableNetworkList failed with error: {ret}")
        return None

    # Cast to WLAN_AVAILABLE_NETWORK_LIST
    net_list_ptr = ctypes.cast(ppAvailableNetworkList, ctypes.POINTER(WLAN_AVAILABLE_NETWORK_LIST))

    num_networks = net_list_ptr.contents.NumberOfItems

    base_ptr = ctypes.addressof(ppAvailableNetworkList.contents) + ctypes.sizeof(WLAN_AVAILABLE_NETWORK_LIST)

    networks = []
    for i in range(num_networks):
        net_ptr = ctypes.cast(base_ptr + i * ctypes.sizeof(WLAN_AVAILABLE_NETWORK), ctypes.POINTER(WLAN_AVAILABLE_NETWORK))
        networks.append(net_ptr.contents)

    wlanapi.WlanFreeMemory(ppAvailableNetworkList)

    return networks

def parse_ssid(network):
    # dot11Ssid is 32 bytes, but the first 4 bytes are length in some definitions,
    # here we must parse it properly.

    # But in our structure, dot11Ssid is c_byte * 32, actually it's a DOT11_SSID struct:
    # According to MSDN, WLAN_AVAILABLE_NETWORK contains DOT11_SSID dot11Ssid;
    # But above we have byte array 32, so let's reinterpret.

    # We reinterpret dot11Ssid bytes as DOT11_SSID struct:
    ssid_bytes = bytes(network.dot11Ssid)
    ssid_struct = DOT11_SSID.from_buffer_copy(ssid_bytes)
    ssid_length = ssid_struct.uSSIDLength
    ssid_str = ssid_struct.ucSSID[:ssid_length].decode('utf-8', errors='ignore')
    return ssid_str

def main():
    client_handle = HANDLE()
    negotiated_version = wintypes.DWORD()

    # Open handle
    ret = wlanapi.WlanOpenHandle(
        2,  # client version for Windows Vista or later
        None,
        ctypes.byref(negotiated_version),
        ctypes.byref(client_handle)
    )
    if ret != ERROR_SUCCESS:
        print(f"WlanOpenHandle failed with error: {ret}")
        sys.exit(1)

    interfaces = get_interface_list(client_handle)
    if not interfaces:
        print("No wireless interfaces found.")
        wlanapi.WlanCloseHandle(client_handle, None)
        sys.exit(1)

    for iface in interfaces:
        print(f"Interface: {iface.strInterfaceDescription}")
        networks = get_available_networks(client_handle, iface.InterfaceGuid)
        if not networks:
            print("  No networks found or failed to get networks.")
            continue

        for net in networks:
            ssid = parse_ssid(net)
            print(f"  SSID: {ssid}, Signal Quality: {net.wlanSignalQuality}%")

    wlanapi.WlanCloseHandle(client_handle, None)

if __name__ == "__main__":
    # Define argtypes and restypes for the functions we use:

    wlanapi.WlanOpenHandle.argtypes = [
        wintypes.DWORD,
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(HANDLE)
    ]
    wlanapi.WlanOpenHandle.restype = wintypes.DWORD

    wlanapi.WlanEnumInterfaces.argtypes = [
        HANDLE,
        wintypes.LPVOID,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
    ]
    wlanapi.WlanEnumInterfaces.restype = wintypes.DWORD

    wlanapi.WlanFreeMemory.argtypes = [wintypes.LPVOID]
    wlanapi.WlanFreeMemory.restype = None

    wlanapi.WlanGetAvailableNetworkList.argtypes = [
        HANDLE,
        ctypes.POINTER(GUID),
        wintypes.DWORD,
        wintypes.LPVOID,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
    ]
    wlanapi.WlanGetAvailableNetworkList.restype = wintypes.DWORD

    wlanapi.WlanCloseHandle.argtypes = [
        HANDLE,
        wintypes.LPVOID
    ]
    wlanapi.WlanCloseHandle.restype = wintypes.DWORD

    main()
