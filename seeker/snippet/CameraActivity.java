//date: 2023-05-30T16:40:28Z
//url: https://api.github.com/gists/1b4e340655fde4517ac769c270630146
//owner: https://api.github.com/users/asengsaragih

package com.suncode.relicbatik.ui.activity;

import android.app.Activity;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.core.content.ContextCompat;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.otaliastudios.cameraview.CameraListener;
import com.otaliastudios.cameraview.CameraView;
import com.otaliastudios.cameraview.PictureResult;
import com.otaliastudios.cameraview.controls.Facing;
import com.otaliastudios.cameraview.controls.Flash;
import com.otaliastudios.cameraview.markers.DefaultAutoFocusMarker;
import com.suncode.relicbatik.R;
import com.suncode.relicbatik.base.BaseActivity;
import com.suncode.relicbatik.base.BaseFunction;
import com.suncode.relicbatik.base.Constant;

import java.io.ByteArrayOutputStream;
import java.io.FileOutputStream;

public class CameraActivity extends BaseActivity implements View.OnClickListener {

    private static final String TAG = "CameraActivityTag";

    CameraView cvDetection;
    FloatingActionButton fabFlip, fabShutter, fabGallery, fabFlash;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);

        init();
        setData();
    }

    private void init() {
        //component
        cvDetection = findViewById(R.id.camera_object_detection);

        fabFlip = findViewById(R.id.fab_object_detection_flip);
        fabShutter = findViewById(R.id.fab_object_detection_shutter);
        fabGallery = findViewById(R.id.fab_object_detection_gallery);
        fabFlash = findViewById(R.id.fab_object_detection_flash);
    }

    private void setData() {
        //camera configuration
        cvDetection.setLifecycleOwner(this);
        cvDetection.setAutoFocusMarker(new DefaultAutoFocusMarker());
        cvDetection.setAutoFocusResetDelay(Long.MAX_VALUE);

        cvDetection.addCameraListener(new CameraListener() {
            @Override
            public void onPictureTaken(@NonNull PictureResult result) {
                super.onPictureTaken(result);

                result.toBitmap(bitmap -> {
                    String image = createImageFromBitmap(bitmap);

                    //intent to detection activity
                    Intent intent = new Intent(CameraActivity.this, DetectionActivity.class);
                    intent.putExtra(Constant.INTENT_BITMAP_FROM_CAMERA, image);

                    startActivity(intent);
                });
            }
        });

        //fab listener
        fabFlip.setOnClickListener(this);
        fabShutter.setOnClickListener(this);
        fabGallery.setOnClickListener(this);
        fabFlash.setOnClickListener(this);
        
        //show dialog info if status info false
        if (!session.getCameraInfoStatus()) {
            showInfo();
        }
    }

    public String createImageFromBitmap(Bitmap bitmap) {
        String fileName = "RELICBATIK_TEMP_IMAGE";//no .png or .jpg needed
        try {
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, bytes);
            FileOutputStream fo = openFileOutput(fileName, Context.MODE_PRIVATE);
            fo.write(bytes.toByteArray());
            // remember close file output
            fo.close();
        } catch (Exception e) {
            e.printStackTrace();
            fileName = null;
        }
        return fileName;
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.fab_object_detection_flip:
                if (cvDetection.getFacing() == Facing.BACK)
                    cvDetection.setFacing(Facing.FRONT);
                else
                    cvDetection.setFacing(Facing.BACK);
                break;
            case R.id.fab_object_detection_shutter:
                cvDetection.takePicture();
                break;
            case R.id.fab_object_detection_gallery:
                openGallery();
                break;
            case R.id.fab_object_detection_flash:
                setFlashCamera();
                break;
        }
    }

    //function for pick image from gallery
    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("image/*");

        mActivityLauncher.launch(Intent.createChooser(intent, "Open"), result -> {
            if (result.getResultCode() == Activity.RESULT_OK) {
                if (result.getData() != null) {
                    uploadImage(result.getData());
                }
            }
        });
    }

    //function for pick image from gallery
    private void uploadImage(Intent data) {
        //convert intent to uri
        Uri uri = data.getData();

        Intent intent = new Intent(CameraActivity.this, DetectionActivity.class);

        intent.putExtra(Constant.INTENT_BITMAP_FROM_GALLERY, uri.toString());

        startActivity(intent);
    }

    //function for set flash camera
    private void setFlashCamera() {
        if (cvDetection.getFlash() == Flash.OFF) {
            //change drawable and turn on flash
            fabFlash.setImageDrawable(ContextCompat.getDrawable(this, R.drawable.ic_camera_flash_on));
            cvDetection.setFlash(Flash.ON);
        } else {
            //change drawable and turn off flash
            fabFlash.setImageDrawable(ContextCompat.getDrawable(this, R.drawable.ic_camera_flash_off));
            cvDetection.setFlash(Flash.OFF);
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.camera_menu, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        switch (item.getItemId()) {
            case R.id.action_info:
                showInfo();
                break;
        }
        return super.onOptionsItemSelected(item);
    }

    //function for show info
    public void showInfo() {
        AlertDialog.Builder builder = new AlertDialog.Builder(CameraActivity.this);

        builder.setTitle("Camera Information");
        builder.setMessage("To getting maximum prediction results, bring the camera closer to the batik and make sure the entire shot is not blurry and the object is not too small");
        builder.setNegativeButton("Close", (dialogInterface, i) -> {
            //save session
            session.setCameraInfoStatus(true);
            dialogInterface.dismiss();
        });

        builder.create().show();
    }
}