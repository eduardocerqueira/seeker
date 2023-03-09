#date: 2023-03-09T16:51:02Z
#url: https://api.github.com/gists/24d866b63031d5ea5f49e658f7643107
#owner: https://api.github.com/users/jorgeluisyh

import json

def show_only_styled_values(featurelayer, layerfile, fieldname):
	"""
	inputs:
		featurelayer : objeto tipo layer
		layerfile: ruta de archivo de tipo .lyr
		fieldname: nombre de campo que se usa para la simbologia tipo texto

	funcion que permite actualizar la simbologia de una capa y solo mostrar las clases que contienen un valor
	"""
	# Convertimos la ruta a un archivo tipo layer
	lyr = arcpy.mapping(layerfile)

	# Creamos una lista de las clasese existentes en el featurelayer
	classes_present = list(set([f[0] for f in arcpy.da.SearchCursor(featurelayer, fieldname)]))

	# Creamos una lista de simbologias vacias 
	simbologias = []
	# obtenemos el diccionario de simbologias del lyr
	sym_dict = json.loads(lyr._arcboject.getsymbology())

	# Recorremos el total de simbologias del lyr y guardamos solo las correspondientes a a las clases existentes
	for simbologia in sym_dict["renderer"]["uniqueValueInfos"]:
		if simbologia["value"] in classes_present:
			simbologias.append(simbologia)

	# Modificamos el diccionario de simbologias del lyr solo para mostrar las clases existentes
	sym_dict["renderer"]["uniqueValueInfos"] = simbologias

	# Actualizamos la simbologia del archivo layer y luego aplicamos los cambios a la capa
	lyr.updateLayerFromJSON(json.dumps({"drawingInfo":sym_dict}))
	arcpy.ApplySymbologyFromLayer_management(featurelayer,lyr)

def modify_field_class(new_field, old_field, lyr):
	lyr = # Layer object, typically from arcpy.mapping.ListLayers (arcpy._mapping.Layer)

	# Example 1a: Modifying single-field renderer using undocumented _arc_object.renderer
	new_field = # New field name for symbolizing layer
	old_field = # Old field name for symbolizing layer 

	sym_xml = lyr._arc_object.renderer
	lyr._arc_object.renderer = sym_xml.replace(old_field, new_field)
	arcpy.RefreshActiveView()

	# Example 1b: Modifying single-field renderer using updateLayerFromJSON
	new_field = # New field name for symbolizing layer (string)

	sym_dict = json.loads(lyr._arc_object.getsymbology())
	sym_dict["renderer"]["field1"] = new_field
	lyr.updateLayerFromJSON(json.dumps({"drawingInfo": sym_dict}))
	arcpy.RefreshActiveView()