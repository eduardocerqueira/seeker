#date: 2024-05-13T16:54:04Z
#url: https://api.github.com/gists/61cda2104f4db4928407999e50d0f467
#owner: https://api.github.com/users/tbbooher

import adsk.core, adsk.fusion, traceback

CIRCLE_DIAMETER = 0.06   # Diameter of the circle in cm (1 mm)
SPACING = 0.4            # Spacing between centers of the circles in cm (5 mm)
OFFSET_DISTANCE = 0.1    # Adjust the offset distance as needed in cm (start with smaller value for testing)

def log_curve_info(curves, ui):
    try:
        for i, curve in enumerate(curves):
            geometry = curve.geometry
            curve_type = type(geometry)
            length = 'N/A'
            if hasattr(geometry, 'evaluator'):
                evaluator = geometry.evaluator
                success, start_param, end_param = evaluator.getParameterExtents()
                if success:
                    success, length = evaluator.getLengthAtParameter(start_param, end_param)
            # ui.messageBox(f"Curve {i}: Type: {curve_type}, Length: {length}")
    except Exception as e:
        ui.messageBox(f"Failed to log curve info:\n{traceback.format_exc()}")

def distance_between_points(point1, point2):
    return ((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)**0.5

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        design = app.activeProduct
        
        selections = ui.activeSelections
        if selections.count == 0:
            ui.messageBox('Please select a profile before running this script.')
            return
        
        selectedEntity = selections.item(0).entity
        if not isinstance(selectedEntity, adsk.fusion.Profile):
            ui.messageBox('Selected entity is not a profile. Please select a profile.')
            return
        
        profile = selectedEntity
        sketch = profile.parentSketch
        curves = adsk.core.ObjectCollection.create()
        for loop in profile.profileLoops:
            for curve in loop.profileCurves:
                curves.add(curve.sketchEntity)

        # Log curve details for debugging
        log_curve_info(curves, ui)

        # Define a manual direction point (e.g., slightly offset from the centroid)
        areaProperties = profile.areaProperties()
        centerPoint = areaProperties.centroid
        dirPoint = adsk.core.Point3D.create(centerPoint.x + 0.1, centerPoint.y, centerPoint.z)
        offsetDistance = OFFSET_DISTANCE

        try:
            offsetCurves = sketch.offset(curves, dirPoint, offsetDistance)
            if not offsetCurves:
                raise RuntimeError("Offset operation did not create any curves.")
        except RuntimeError as e:
            ui.messageBox(f"Error during offsetting: {str(e)}\nCurves Count: {curves.count}")
            return
        
        # Set offset curves as construction lines and draw circles
        circleDiameter = CIRCLE_DIAMETER
        spacing = SPACING
        existingPoints = []

        for curve in offsetCurves:
            curve.isConstruction = True
            if hasattr(curve.geometry, 'evaluator'):
                curveEvaluator = curve.geometry.evaluator
                success, startParam, endParam = curveEvaluator.getParameterExtents()
                if success:
                    success, curveLength = curveEvaluator.getLengthAtParameter(startParam, endParam)
                    if success:
                        numCircles = int(curveLength / spacing) + 1
                        for i in range(numCircles):
                            length = i * spacing
                            if length > curveLength:
                                length = curveLength
                            success, param = curveEvaluator.getParameterAtLength(startParam, length)
                            if success:
                                success, point = curveEvaluator.getPointAtParameter(param)
                                if success:
                                    too_close = False
                                    for existingPoint in existingPoints:
                                        if distance_between_points(point, existingPoint) < (spacing+0.01):
                                            too_close = True
                                            break
                                    if not too_close:
                                        sketch.sketchCurves.sketchCircles.addByCenterRadius(point, circleDiameter / 2)
                                        existingPoints.append(point)
            
    except Exception as e:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))