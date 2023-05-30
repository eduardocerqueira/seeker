//date: 2023-05-30T16:40:28Z
//url: https://api.github.com/gists/1b4e340655fde4517ac769c270630146
//owner: https://api.github.com/users/asengsaragih

package com.suncode.relicbatik.base;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.EditText;
import android.widget.Toast;

import androidx.activity.result.ActivityResult;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import com.google.android.gms.auth.api.signin.GoogleSignIn;
import com.google.android.gms.auth.api.signin.GoogleSignInAccount;
import com.google.android.gms.auth.api.signin.GoogleSignInClient;
import com.google.android.gms.auth.api.signin.GoogleSignInOptions;
import com.google.firebase.auth.FirebaseAuth;
import com.suncode.relicbatik.R;
import com.suncode.relicbatik.api.ApiClient;
import com.suncode.relicbatik.api.ApiService;
import com.suncode.relicbatik.helper.ActivityResultHelper;

public class BaseActivity extends AppCompatActivity {

    protected AppCompatActivity mActivity;
    protected Session session;
    private AlertDialog mLoadingDialog;

    //fungsi untuk startactivity result
    protected ActivityResultHelper<Intent, ActivityResult> mActivityLauncher = ActivityResultHelper.registerActivityForResult(this);

    //firebase
    protected GoogleSignInClient mGoogleSignInClient;
    protected GoogleSignInAccount mGoogleSignInAccount;
    protected FirebaseAuth mAuth;

    //for api public
    protected ApiService mApiService;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        mActivity = this;

        //intialize session
        session = new Session(this);

        // Check for existing Google Sign In account, if the user is already signed in
        // the GoogleSignInAccount will be non-null.
        mGoogleSignInAccount = GoogleSignIn.getLastSignedInAccount(this);

        // Configure Google Sign In
        GoogleSignInOptions gso = new GoogleSignInOptions.Builder(GoogleSignInOptions.DEFAULT_SIGN_IN)
                .requestIdToken(getString(R.string.default_web_client_id))
                .requestEmail()
                .requestProfile()
                .build();

        // Build a GoogleSignInClient with the options specified by gso.
        mGoogleSignInClient = GoogleSignIn.getClient(this, gso);

        //signInWithCredential
        mAuth = FirebaseAuth.getInstance();

        //api service
        mApiService = ApiClient.builder().create(ApiService.class);

        //inisialisasi loding bar
        loadingInitialize();
    }

    private void loadingInitialize() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        View view= LayoutInflater.from(this).inflate(R.layout.progress_loading, null);
        builder.setView(view);
        builder.setCancelable(false);
        mLoadingDialog = builder.create();
    }

    protected void showLoading (boolean show){
        if (show)
            mLoadingDialog.show();
        else
            mLoadingDialog.dismiss();
    }

    protected void toast(String message) {
        //function show toast on other activity
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show();
    }

    protected void longToast(String message) {
        //function show toast on other activity
        Toast.makeText(this, message, Toast.LENGTH_LONG).show();
    }

    protected boolean isEditTextEmpty(EditText editText) {
        if (editText.getText().toString().equals(" "))
            return true;
        else
            return TextUtils.isEmpty(editText.getText());
    }

    protected AlertDialog.Builder dialogMessage(Context context, String title, String message) {
        //alert dialog builder
        return new AlertDialog.Builder(context).setTitle(title).setMessage(message);
    }

    //function for check read storage permission
    protected boolean isReadStroragePermissionGranted() {
        //check permission access read storage
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU)
            return checkSelfPermission(Manifest.permission.READ_MEDIA_IMAGES) == PackageManager.PERMISSION_GRANTED;
        else
            return checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
    }

    //function for check camera permission
    protected boolean isCameraPermissionGranted() {
        return checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }

    //function for check camera permission
    protected boolean isAudioPermissionGranted() {
        return checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED;
    }

    //function for check camera permission
    protected boolean isCallPhonePermissionGranted() {
        return checkSelfPermission(Manifest.permission.CALL_PHONE) == PackageManager.PERMISSION_GRANTED;
    }

    //function for check camera permission
    protected boolean isLocationPermissionGranted() {
        return checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED
                && checkSelfPermission(Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED;
    }

    //function for get permission camera
    protected void getPermissionLocation() {
        ActivityCompat.requestPermissions(this, new String[] {
                Manifest.permission.ACCESS_FINE_LOCATION,
                Manifest.permission.ACCESS_COARSE_LOCATION
        }, Constant.REQUEST_PERMISSION_LOCATION);
    }

    //function for get permission read storage
    protected void getPermissionReadStorage() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU)
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.READ_MEDIA_IMAGES, Manifest.permission.WRITE_EXTERNAL_STORAGE}, Constant.REQUEST_PERMISSION_READ_STORAGE);
        else
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE}, Constant.REQUEST_PERMISSION_READ_STORAGE);
    }

    //function for get permission camera
    protected void getPermissionCamera() {
        ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, Constant.REQUEST_PERMISSION_CAMERA);
    }

    //function for get permission camera
    protected void getPermissionCallPhone() {
        ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CALL_PHONE}, Constant.REQUEST_PERMISSION_CALL_PHONE);
    }

    //function for get permission camera
    protected void getPermissionAudio() {
        ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.RECORD_AUDIO}, Constant.REQUEST_PERMISSION_AUDIO);
    }

    //function for getting permission when click camera button in main first time
    protected void getPermissionCameraAudioAndReadStorage() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            ActivityCompat.requestPermissions(
                    this,
                    new String[] {
                            Manifest.permission.READ_MEDIA_IMAGES,
                            Manifest.permission.WRITE_EXTERNAL_STORAGE,
                            Manifest.permission.CAMERA,
                            Manifest.permission.RECORD_AUDIO,
                    },
                    Constant.REQUEST_PERMISSION_FIRST_TIME
            );
        } else {
            ActivityCompat.requestPermissions(
                    this,
                    new String[] {
                            Manifest.permission.READ_EXTERNAL_STORAGE,
                            Manifest.permission.WRITE_EXTERNAL_STORAGE,
                            Manifest.permission.CAMERA,
                            Manifest.permission.RECORD_AUDIO,
                    },
                    Constant.REQUEST_PERMISSION_FIRST_TIME
            );
        }
    }
}
