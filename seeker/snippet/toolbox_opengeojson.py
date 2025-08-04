#date: 2025-08-04T17:08:45Z
#url: https://api.github.com/gists/0a8fa2c632afaa20a535becabadbc7ca
#owner: https://api.github.com/users/programandaana

#Script for Processing in QGIS 
from qgis.core import (QgsProcessingAlgorithm, 
                       QgsProcessingParameterFile,
                       QgsProcessingOutputVectorLayer,
                       QgsProcessingException,
                       QgsProject,
                       QgsVectorLayer)
from qgis.PyQt.QtCore import QCoreApplication
import os

class LoadGeoJSONAlgorithm(QgsProcessingAlgorithm):
    """
    This algorithm loads a GeoJSON file into QGIS.
    """
    
    # Constants used to refer to parameters and outputs
    INPUT = 'INPUT'
    OUTPUT = 'OUTPUT'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return LoadGeoJSONAlgorithm()

    def name(self):
        return 'loadgeojson'

    def displayName(self):
        return self.tr('Load GeoJSON File')

    def group(self):
        return self.tr('Utilities')

    def groupId(self):
        return 'utilities'

    def shortHelpString(self):
        return self.tr("Loads a GeoJSON file into QGIS as a new vector layer.")

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT,
                self.tr('Input GeoJSON file'),
                behavior=QgsProcessingParameterFile.File,
                fileFilter='GeoJSON files (*.geojson *.json)',
                defaultValue=None
            )
        )
        
        self.addOutput(
            QgsProcessingOutputVectorLayer(
                self.OUTPUT,
                self.tr('Loaded GeoJSON layer')
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        file_path = self.parameterAsFile(parameters, self.INPUT, context)
        
        # Generate layer name from file path
        layer_name = os.path.splitext(os.path.basename(file_path))[0]
        
        layer = QgsVectorLayer(file_path, layer_name, 'ogr')
        
        if not layer.isValid():
            raise QgsProcessingException('Failed to load GeoJSON file!')
        
        QgsProject.instance().addMapLayer(layer)
        
        return {self.OUTPUT: layer.id()}