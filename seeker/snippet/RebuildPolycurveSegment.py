#date: 2022-03-16T17:10:19Z
#url: https://api.github.com/gists/be2460317a254189157726234d26864a
#owner: https://api.github.com/users/pgolay

import Rhino
import rhinoscriptsyntax as rs
import scriptcontext as sc
import System

def RebuildPolycurveSegment():
    def filter_polycurve(rhino_object, geometry, component_index):
        if  isinstance(geometry, Rhino.Geometry.PolyCurve):
            return True
        return False
    while True:
        rs.UnselectAllObjects()
        color = Rhino.ApplicationSettings.AppearanceSettings.FeedbackColor
        deviation_color = System.Drawing.Color.Red
        while True:
            deg = 3
            count = 4
            delete = True
            tan = True
            
            if sc.sticky.has_key('REBUILD_DEG'):
                deg = sc.sticky['REBUILD_DEG']
            
            if sc.sticky.has_key('REBUILD_COUNT'):
                count = sc.sticky['REBUILD_COUNT']
                
            if count < deg+1:
                count = deg+1
                sc.sticky['REBUILD_COUNT']  = count
                
            if sc.sticky.has_key('RBLD_DELETE'):
                delete = sc.sticky['RBLD_DELETE']
                
            if sc.sticky.has_key('RBLD_TAN'):
                tan = sc.sticky['RBLD_TAN']
    
                
            go = Rhino.Input.Custom.GetObject()
            go.GeometryFilter = Rhino.DocObjects.ObjectType.Curve
            go.SetCustomGeometryFilter (filter_polycurve)
            
            opDeg = Rhino.Input.Custom.OptionInteger(deg, 1, 11)
            opCount = Rhino.Input.Custom.OptionInteger(count, True, 2 )
            opDel = Rhino.Input.Custom.OptionToggle(delete, 'No', 'Yes')
            opTan = Rhino.Input.Custom.OptionToggle(tan, 'No', 'Yes')
            
            go.AddOptionInteger('CurveDegree', opDeg)
            go.AddOptionInteger('PointCount', opCount)
            go.AddOptionToggle('ReplaceSegment', opDel)
            go.AddOptionToggle('PreserveTangents', opTan)
            
            go.AcceptNothing (True)
            go.AcceptNumber(True, False)
            
            rc = go.Get()
            
            if go.CommandResult()!=Rhino.Commands.Result.Success:
                #conduit.Enabled=False
                return
            if rc == Rhino.Input.GetResult.Object:
                oref = go.Object(0)
                id = oref.ObjectId
                crv = oref.Geometry()
                par = crv.ClosestPoint(oref.SelectionPoint())[1]
                break
                
            if rc == Rhino.Input.GetResult.Option:
                idx = go.OptionIndex()
                print idx
                if idx ==1:
                    deg = opDeg.CurrentValue
                    if count < deg+1:
                        count = deg+1
                elif idx ==2:
                    count = opCount.CurrentValue
                    if deg > count -1:
                        deg = count-1
    
                delete = opDel.CurrentValue
                tan = opTan.CurrentValue
                sc.sticky['REBUILD_DEG'] = deg
                sc.sticky['REBUILD_COUNT']  = count
                sc.sticky['RBLD_DELETE'] = delete
                sc.sticky['RBLD_TAN'] = tan
                #conduit.Enabled=False
            elif rc == Rhino.Input.GetResult.Number:
                num = go.Number()
                if 0< int(num) < 12:
                    deg = int(num)
                    sc.sticky['REBUILD_DEG'] = deg
                    #conduit.Enabled=False
                    continue
            
            elif rc == Rhino.Input.GetResult.Nothing:
                #conduit.Enabled=False
                return
                
            elif rc == Rhino.Input.GetResult.Cancel:
                print('Cancelled')
                #conduit.Enabled=False
                return
    
        segs = [crv.SegmentCurve(n) for n in range(crv.SegmentCount)]
     
        idx = crv.SegmentIndex(par)
        seg = crv.SegmentCurve(idx)
        
        #/////////////Rebuild
        seg =  seg.Rebuild(count,deg,True)
        #/////////////
        if not delete:
            sc.doc.Objects.AddCurve(seg)
            sc.doc.Views.Redraw()

        else:
            segs[idx] = seg
        
            c2= Rhino.Geometry.PolyCurve()
            for seg in segs:
                c2.AppendSegment(seg)
            c2.CleanUp()
            sc.doc.Objects.Replace(id,c2)
            sc.doc.Views.Redraw()

if __name__ == '__main__':RebuildPolycurveSegment()
