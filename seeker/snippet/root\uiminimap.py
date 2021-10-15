#date: 2021-10-15T17:15:14Z
#url: https://api.github.com/gists/53e9f09857822330f89046ca549cbc00
#owner: https://api.github.com/users/Iicymnius

#find;

		self.AtlasShowButton = 0

#add below;

		if app.ENABLE_DUNGEON_INFO_SYSTEM:
			self.dungeonInfoButton = 0

#find again;

			self.serverInfo = self.GetChild("ServerInfo")

#add below;

			if app.ENABLE_DUNGEON_INFO_SYSTEM:
				self.dungeonInfoButton = self.GetChild("l_button")

#find again;

		self.MiniMapShowButton.SetEvent(ui.__mem_func__(self.ShowMiniMap))

#add below;

		if app.ENABLE_DUNGEON_INFO_SYSTEM:
			self.dungeonInfoButton.SetEvent(ui.__mem_func__(self.DungeonInfo))

#find again;

	def ShowAtlas(self):
		if not miniMap.IsAtlas():
			return
		if not self.AtlasWindow.IsShow():
			self.AtlasWindow.Show()

#add below;

	if app.ENABLE_DUNGEON_INFO_SYSTEM:
		def DungeonInfo(self):
			if self.interface:
				self.interface.ShowDungeonInfoInterface()