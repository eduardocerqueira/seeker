#date: 2023-09-29T17:01:52Z
#url: https://api.github.com/gists/85636f00830026761e97acbf72848a92
#owner: https://api.github.com/users/josephyooo

# merge two configs as if merging two folders and a overrides existing items in b (i.e. the items are options)

a = """version:3465
autoJump:false
operatorItemsTab:true
snooperEnabled:false
darkMojangStudiosBackground:true
simulationDistance:6
guiScale:3
maxFps:260
resourcePacks:["vanilla","fabric","continuity:glass_pane_culling_fix","continuity:default","file/SodiumTranslations.zip","file/Mod Menu Helper.zip","file/Chat Reporting Helper.zip","file/Fast Better Grass.zip"]
incompatibleResourcePacks:[]
advancedItemTooltips:true
tutorialStep:none
skipMultiplayerWarning:true
joinedFirstServer:true
telemetryOptInExtra:false
onboardAccessibility:false
key_key.saveToolbarActivator:key.keyboard.unknown
key_zoomify.key.zoom:key.keyboard.c
key_zoomify.key.zoom.secondary:key.keyboard.unknown
key_key.antighost.reveal:key.keyboard.unknown
key_iris.keybind.reload:key.keyboard.unknown
key_iris.keybind.toggleShaders:key.keyboard.unknown
key_iris.keybind.shaderPackSelection:key.keyboard.unknown
key_key.fabricskyboxes.toggle:key.keyboard.unknown
"""
b = """version:3465
autoJump:false
operatorItemsTab:false
autoSuggestions:true
chatColors:true
chatLinks:true
chatLinksPrompt:true
enableVsync:true
entityShadows:false
forceUnicodeFont:false
discrete_mouse_scroll:false
invertYMouse:false
realmsNotifications:true
reducedDebugInfo:false
showSubtitles:false
directionalAudio:false
touchscreen:false
fullscreen:true
bobView:false
toggleCrouch:false
toggleSprint:false
darkMojangStudiosBackground:false
hideLightningFlashes:false
mouseSensitivity:0.5042583626760564
fov:1.0
screenEffectScale:0.20825264084507025
fovEffectScale:0.0
darknessEffectScale:1.0
glintSpeed:0.10018705985915492
glintStrength:0.75
damageTiltStrength:1.0
highContrast:false
gamma:1.0
renderDistance:32
simulationDistance:12
entityDistanceScaling:1.0
guiScale:3
particles:1
maxFps:60
graphicsMode:0
ao:false
prioritizeChunkUpdates:0
biomeBlendRadius:0
renderClouds:"false"
resourcePacks:[]
incompatibleResourcePacks:[]
lastServer:
lang:en_us
soundDevice:""
chatVisibility:0
chatOpacity:1.0
chatLineSpacing:0.0
textBackgroundOpacity:0.5
backgroundForChatOnly:true
hideServerAddress:false
advancedItemTooltips:false
pauseOnLostFocus:true
overrideWidth:0
overrideHeight:0
chatHeightFocused:1.0
chatDelay:0.0
chatHeightUnfocused:0.4375
chatScale:1.0
chatWidth:1.0
notificationDisplayTime:1.0
mipmapLevels:0
useNativeTransport:true
mainHand:"right"
attackIndicator:1
narrator:0
tutorialStep:none
mouseWheelSensitivity:1.0
rawMouseInput:true
glDebugVerbosity:1
skipMultiplayerWarning:true
skipRealms32bitWarning:false
hideMatchedNames:true
joinedFirstServer:true
hideBundleTutorial:false
syncChunkWrites:true
showAutosaveIndicator:true
allowServerListing:true
onlyShowSecureChat:false
panoramaScrollSpeed:1.0
telemetryOptInExtra:false
onboardAccessibility:false
key_key.attack:key.mouse.left
key_key.use:key.mouse.right
key_key.forward:key.keyboard.w
key_key.left:key.keyboard.a
key_key.back:key.keyboard.s
key_key.right:key.keyboard.d
key_key.jump:key.keyboard.space
key_key.sneak:key.keyboard.left.shift
key_key.sprint:key.keyboard.left.control
key_key.drop:key.keyboard.q
key_key.inventory:key.keyboard.e
key_key.chat:key.keyboard.t
key_key.playerlist:key.keyboard.tab
key_key.pickItem:key.mouse.middle
key_key.command:key.keyboard.slash
key_key.socialInteractions:key.keyboard.p
key_key.screenshot:key.keyboard.f2
key_key.togglePerspective:key.mouse.5
key_key.smoothCamera:key.keyboard.unknown
key_key.fullscreen:key.keyboard.f11
key_key.spectatorOutlines:key.keyboard.unknown
key_key.swapOffhand:key.keyboard.f
key_key.saveToolbarActivator:key.keyboard.c
key_key.loadToolbarActivator:key.keyboard.x
key_key.advancements:key.keyboard.l
key_key.hotbar.1:key.keyboard.1
key_key.hotbar.2:key.keyboard.2
key_key.hotbar.3:key.keyboard.3
key_key.hotbar.4:key.keyboard.4
key_key.hotbar.5:key.keyboard.5
key_key.hotbar.6:key.keyboard.6
key_key.hotbar.7:key.keyboard.7
key_key.hotbar.8:key.keyboard.8
key_key.hotbar.9:key.keyboard.9
soundCategory_master:1.0
soundCategory_music:0.26463468309859156
soundCategory_record:1.0
soundCategory_weather:1.0
soundCategory_block:1.0
soundCategory_hostile:1.0
soundCategory_neutral:1.0
soundCategory_player:1.0
soundCategory_ambient:1.0
soundCategory_voice:1.0
modelPart_cape:true
modelPart_jacket:true
modelPart_left_sleeve:true
modelPart_right_sleeve:true
modelPart_left_pants_leg:true
modelPart_right_pants_leg:true
modelPart_hat:true
"""

asp = a.split("\n")
bsp = b.split("\n")

for line in asp:
  s = line.split(":")
  if s[0] in b:
    bsp[b[:b.index(s[0])].count("\n")] = line
  else:
    bsp.append(line)

print("\n".join(bsp))
