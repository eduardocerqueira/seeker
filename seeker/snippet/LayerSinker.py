#date: 2023-10-12T17:09:59Z
#url: https://api.github.com/gists/a1f78410e547b2acc85f1ef078eb9e93
#owner: https://api.github.com/users/connordavenport

from mojo.subscriber import Subscriber, registerRoboFontSubscriber

class LayerSinker(Subscriber):

	def fontDocumentDidOpen(self,info):
		font = info["font"]
		font.lib['com.typemytype.robofont.syncGlyphLayers'] = 'metrics'

registerRoboFontSubscriber(LayerSinker)