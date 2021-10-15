#date: 2021-10-15T17:15:03Z
#url: https://api.github.com/gists/38d6604143aa5ccf608d2e63cf85553c
#owner: https://api.github.com/users/Iicymnius

#add;

if app.ENABLE_DUNGEON_INFO_SYSTEM:
	import uiDungeonInfo

#find;

		self.wndGuild = None

#add below;

		if app.ENABLE_DUNGEON_INFO_SYSTEM:
			self.wndDungeonInfo = None

#find again;

	def __MakeHelpWindow(self):

#add above;

		if app.ENABLE_DUNGEON_INFO_SYSTEM:
			self.wndDungeonInfo = uiDungeonInfo.DungeonInfo()
		else:
			self.wndDungeonInfo = None

#find again;

		if self.wndDragonSoulRefine:
			self.wndDragonSoulRefine.Destroy()

#add below;

		if app.ENABLE_DUNGEON_INFO_SYSTEM:
			if self.wndDungeonInfo:
				self.wndDungeonInfo.Hide()
				self.wndDungeonInfo.Destroy()
				self.wndDungeonInfo = None
				del self.wndDungeonInfo

#find again;

	def ShowMouseImage(self):

#add above;

		if app.ENABLE_DUNGEON_INFO_SYSTEM:
			if self.wndDungeonInfo:
				self.wndDungeonInfo.Hide()

#find again;

		hideWindows = filter(lambda x:x.IsShow(), hideWindows)

#add above;

		if app.ENABLE_DUNGEON_INFO_SYSTEM:
			if self.wndDungeonInfo:
				hideWindows += self.wndDungeonInfo,

#add to bottom;

	if app.ENABLE_DUNGEON_INFO_SYSTEM:
		def ShowDungeonInfoInterface(self):
			if app.ENABLE_PLAYER_SECURITY_SYSTEM and player.IsSecurityActivate():
				return
			if False == self.wndDungeonInfo.IsShow():
				self.wndDungeonInfo.Open()
			else:
				self.wndDungeonInfo.Close()