//date: 2022-01-06T17:18:57Z
//url: https://api.github.com/gists/f6c5c7654ea7c28aa73ae532b9708baa
//owner: https://api.github.com/users/jcfandino

package com.example;

import com.jme3.app.SimpleApplication;
import com.jme3.asset.plugins.ClasspathLocator;
import com.jme3.collision.CollisionResult;
import com.jme3.collision.CollisionResults;
import com.jme3.font.BitmapText;
import com.jme3.light.AmbientLight;
import com.jme3.material.Material;
import com.jme3.math.ColorRGBA;
import com.jme3.math.Ray;
import com.jme3.math.Vector2f;
import com.jme3.renderer.queue.RenderQueue.Bucket;
import com.jme3.scene.Geometry;
import com.jme3.scene.VertexBuffer.Type;
import com.jme3.scene.shape.Quad;

public class TestTexturePickApp extends SimpleApplication {

  public static void main(String[] args) {
    new TestTexturePickApp().start();
  }

  private Material boxMat;

  @Override
  public void simpleInitApp() {
    // cam config
    flyCam.setMoveSpeed(20f);
    flyCam.setRotationSpeed(2f);
    // add level
    assetManager.registerLocator("levels", ClasspathLocator.class);
    var map = assetManager.loadModel("levels/map1.j3o");
    map.setQueueBucket(Bucket.Transparent);
    rootNode.attachChild(map);
    rootNode.addLight(new AmbientLight(ColorRGBA.White));
    // ui
    createCrosshair();
    createColorBox();
  }

  private void createCrosshair() {
    var font = assetManager.loadFont("Interface/Fonts/Default.fnt");
    var text = new BitmapText(font, false);
    text.setSize(font.getCharSet().getRenderedSize());
    text.setColor(ColorRGBA.White);
    text.setText("+");
    text.setLocalTranslation((cam.getWidth() - text.getLineWidth()) * 0.5f,
        (cam.getHeight() + text.getLineHeight()) * 0.5f, 0);
    guiNode.attachChild(text);
  }

  private void createColorBox() {
    boxMat = new Material(assetManager, "Common/MatDefs/Misc/Unshaded.j3md");
    boxMat.setColor("Color", ColorRGBA.Black);
    var quad = new Geometry("box", new Quad(40, 40));
    quad.setMaterial(boxMat);
    quad.setLocalTranslation(cam.getWidth() * 0.5f - 20, cam.getHeight() * 0.5f - 60, 0);
    guiNode.attachChild(quad);
  }

  @Override
  public void simpleUpdate(float tpf) {
    var results = new CollisionResults();
    // Aim the ray from camera location in camera direction
    // (assuming crosshairs in center of screen).
    var ray = new Ray(cam.getLocation(), cam.getDirection());
    // Collect intersections between ray and all nodes in results list.
    rootNode.collideWith(ray, results);
    // Use the results
    if (results.size() > 0) {
      // The closest result is the target that the player picked:
      updateBoxColor(results.getClosestCollision());
    }
  }

  private void updateBoxColor(CollisionResult result) {
    var geometry = result.getGeometry();
    var material = geometry.getMaterial();
    if (material.getTextureParam("LightMap") != null) {
      // https://answers.unity.com/questions/383804/calculate-uv-coordinates-of-3d-point-on-plane-of-m.html
      var triangle = result.getTriangle(null);
      var p1 = triangle.get1();
      var p2 = triangle.get2();
      var p3 = triangle.get3();
      // calculate vectors from point f to vertices p1, p2 and p3:
      var point = result.getContactPoint();
      var f1 = p1.subtract(point);
      var f2 = p2.subtract(point);
      var f3 = p3.subtract(point);
      // calculate the areas and factors (order of parameters doesn't matter):
      // a = || (p1-p2) x (p1-p3) ||
      var a = p1.subtract(p2).cross(p1.subtract(p3)).length(); // main triangle area a
      var a1 = f2.cross(f3).length() / a; // p1's triangle area / a
      var a2 = f3.cross(f1).length() / a; // p2's triangle area / a
      var a3 = f1.cross(f2).length() / a; // p3's triangle area / a

      // find the uv corresponding to point f
      // (uv1/uv2/uv3 are associated to p1/p2/p3):
      var indices = new int[3];
      var mesh = geometry.getMesh();
      mesh.getTriangle(result.getTriangleIndex(), indices);
      var buffer = mesh.getBuffer(Type.TexCoord2);
      var uv1 = new Vector2f((float) buffer.getElementComponent(indices[0], 0),
          (float) buffer.getElementComponent(indices[0], 1));
      var uv2 = new Vector2f((float) buffer.getElementComponent(indices[1], 0),
          (float) buffer.getElementComponent(indices[1], 1));
      var uv3 = new Vector2f((float) buffer.getElementComponent(indices[2], 0),
          (float) buffer.getElementComponent(indices[2], 1));

      // uv = uv1 * a1 + uv2 * a2 + uv3 * a3
      var uv = uv1.mult(a1).add(uv2.mult(a2)).add(uv3.mult(a3));

      // get pixel color from lightmap
      var lightmap = material.getTextureParam("LightMap");
      var image = lightmap.getTextureValue().getImage();
      var width = image.getWidth();
      var height = image.getHeight();
      var x = (int) (uv.x * width);
      var y = (int) (uv.y * height);
      // assumes ABGR8
      var data = image.getData(0);
      data.position(4 * (y * width + x));
      var o = ((float) Byte.toUnsignedInt(data.get())) / 255f;
      var b = ((float) Byte.toUnsignedInt(data.get())) / 255f;
      var g = ((float) Byte.toUnsignedInt(data.get())) / 255f;
      var r = ((float) Byte.toUnsignedInt(data.get())) / 255f;
      var color = new ColorRGBA(r, g, b, o);
      System.out.println(String.format("%4d x %4d => %f %f %f - %f", x, y, r, g, b, o));
      boxMat.setColor("Color", color);
    } else {
      boxMat.setColor("Color", ColorRGBA.randomColor());
    }
  }

}
