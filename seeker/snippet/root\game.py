#date: 2021-10-15T17:14:53Z
#url: https://api.github.com/gists/4bd3bcf1d861e4cb07ced86fa5733d05
#owner: https://api.github.com/users/Iicymnius

#find;

	def BINARY_ServerCommand_Run(self, line):

#add above;

		if app.ENABLE_DUNGEON_INFO_SYSTEM:
			self.serverCommander.SAFE_RegisterCallBack("CleanDungeonInfo", self.CleanDungeonInfo)
			self.serverCommander.SAFE_RegisterCallBack("UpdateDungeonInfo", self.UpdateDungeonInfo)

#add to bottom;

	if app.ENABLE_DUNGEON_INFO_SYSTEM:
		def CleanDungeonInfo(self):
			import constInfo
			constInfo.dungeonInfo = []

		def UpdateDungeonInfo(self, type, organization, minLevel, partyMembers, mapIndex, mapName, mapEntrance, mapCoordX, mapCoordY, cooldown, duration, maxLevel, strengthBonus, resistanceBonus, itemVnum, bossVnum):
			type = int(type)
			organization = int(organization)
			minLevel = int(minLevel)
			partyMembers = int(partyMembers)
			mapName = str(mapName).replace("_", " ")
			mapEntrance = str(mapEntrance).replace("_", " ")
			mapIndex = int(mapIndex)
			mapCoordX = int(mapCoordX)
			mapCoordY = int(mapCoordY)
			cooldown = int(cooldown)
			duration = int(duration)
			maxLevel = int(maxLevel)
			strengthBonus = int(strengthBonus)
			resistanceBonus = int(resistanceBonus)
			itemVnum = int(itemVnum)

			constInfo.dungeonInfo.append(\
				{
					"type" : type,\
					"organization" : organization,\
					"min_level" : minLevel,\
					"party_members" : partyMembers,\
					"map" : mapName,\
					"entrance_map" : mapEntrance,\
					"map_index" : mapIndex,\
					"map_coord_x" : mapCoordX,\
					"map_coord_y" : mapCoordY,\
					"cooldown" : cooldown,\
					"duration" : duration,\
					"max_level" : maxLevel,\
					"strength_bonus" : strengthBonus,\
					"resistance_bonus" : resistanceBonus,\
					"item_vnum" : itemVnum,\
					"boss_vnum" : bossVnum,\
				},
			)