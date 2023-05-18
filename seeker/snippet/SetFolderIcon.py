#date: 2023-05-18T17:08:09Z
#url: https://api.github.com/gists/0eaaf41dd4ccf923ba6d78b0759435ce
#owner: https://api.github.com/users/kubinka0505

"""SetFolderIcon

Set folder's icon on Windows from any DLL resource."""
import ctypes
from os.path import *
from ctypes.wintypes import BYTE, WORD, DWORD, LPWSTR
from ctypes import POINTER, Structure, c_wchar, c_int, sizeof, byref

__author__	= ["Christoph Gohlke", "kubinka0505"]
__credits__	= __author__
__date__	= "18.05.2023"

def SetFolderIcon(folder: abspath, resource: abspath = r"C:\Windows\System32\ImageRes.dll", index: int = 1) -> tuple:
	"""Set folder icon."""

	class FolderCustomSettings(Structure):
		class GUID(Structure):
			_fields_ = [
				("Data1", DWORD),
				("Data2", WORD),
				("Data3", WORD),
				("Data4", BYTE * 8)
			]

		_fields_ = [
			("dwSize", DWORD),
			("dwMask", DWORD),
			("pvid", POINTER(GUID)),
			("pszWebViewTemplate", LPWSTR),
			("cchWebViewTemplate", DWORD),
			("pszWebViewTemplateVersion", LPWSTR),
			("pszInfoTip", LPWSTR),
			("cchInfoTip", DWORD), 
			("pclsid", POINTER(GUID)),
			("dwFlags", DWORD),
			("pszIconFile", LPWSTR),
			("cchIconFile", DWORD),
			("iIconIndex", c_int),
			#("pszLogo", LPWSTR),
			#("cchLogo", DWORD)
		]

	class FileInfo(Structure):
		_fields_ = [
			("hIcon", c_int),
			("iIcon", c_int),
			#("dwAttributes", DWORD),
			("szDisplayName", c_wchar * 260),
			#("szTypeName", c_wchar * 80),
		]

	#-=-=-=-#

	Folder = abspath(expanduser(expandvars(folder.replace("/", sep))))
	Resource = abspath(expanduser(expandvars(resource.replace("/", sep))))
	Index = min(max(-8**8, abs(index)), 8**8)
	
	if splitext(Resource)[-1].upper()[1:] != "DLL":
		raise NotImplementedError(f'"{Extension}" icon format')

	if not exists(Folder):
		raise FileNotFoundError(Folder)
	elif isfile(Folder):
		raise TypeError(f'Is file ("{basename(Folder)}")')

	if not exists(Resource):
		Resource = r"C:\Windows\System32\Shell32.dll"
	elif isdir(Resource):
		raise TypeError(f'Is folder ("{basename(Resource)}")')

	#-=-=-=-#

	SFI = FileInfo()
	Shell32 = ctypes.windll.shell32

	FCS = FolderCustomSettings()
	FCS.dwSize = sizeof(FCS)
	FCS.dwMask = 16
	FCS.pszIconFile = resource
	FCS.cchIconFile = 0
	FCS.iIconIndex = index

	Shell32.SHGetSetFolderCustomSettings(byref(FCS), Folder, 2)
	Index_ = Shell32.Shell_GetCachedImageIndexW(SFI.szDisplayName, SFI.iIcon, 0)
	Shell32.SHUpdateImageW(SFI.szDisplayName, SFI.iIcon, 0, Index)

	return Folder, Resource, Index

import random
print(SetFolderIcon("teset", "C:\Windows\System32\ImageRes.dll", index = 77))