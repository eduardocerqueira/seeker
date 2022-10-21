//date: 2022-10-21T17:21:02Z
//url: https://api.github.com/gists/8a86298141b7b41215083574f5ae7bad
//owner: https://api.github.com/users/thecodecafe

package com.sendsolid;

import com.facebook.react.bridge.Callback;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.ReadableNativeMap;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.bridge.WritableNativeMap;
import com.verygoodsecurity.vgscollect.core.Environment;
import com.verygoodsecurity.vgscollect.core.VGSCollect;
import com.verygoodsecurity.vgscollect.core.HTTPMethod;
import com.verygoodsecurity.vgscollect.core.model.network.VGSRequest;
import com.verygoodsecurity.vgscollect.core.model.network.VGSResponse;
import com.verygoodsecurity.vgscollect.core.VgsCollectResponseListener;

import java.util.HashMap;

class SendSolidModule extends ReactContextBaseJavaModule {
    SendSolidModule(ReactApplicationContext context) {
        super(context);
    }
    
    @Override
    public String getName() {
        return "SendSolidModule";
    }

    @ReactMethod
    void linkUSDebitCard(final ReadableMap params, final Callback callback) {
        // get the selected environment
        Environment environment = params.getString("environment") == "live" ? Environment.LIVE : Environment.SANDBOX;
        // initialize VGS collect
        VGSCollect vgsForm = new VGSCollect.Builder(this.getCurrentActivity(), params.getString("vaultId"))
                .setEnvironment(environment)
                .create();

        // setting sd-pin-token as a custom header
        HashMap<String, String> header = new HashMap<>();
        header.put("sd-debitcard-token", params.getString("cardToken"));
        vgsForm.setCustomHeaders(header);

        // setting custom data using the user entered parameters
        HashMap<String, HashMap<String, Object>> data = new HashMap<>();
        HashMap<String, Object> debitCard = new HashMap<>();
        debitCard.put("expiryMonth", params.getString("expMonth"));
        debitCard.put("expiryYear", params.getString("expYear"));
        debitCard.put("cardNumber", params.getString("cardNumber"));
        debitCard.put("cvv", params.getString("cvv"));

        //address is hardcoded this will be the address of the contact
        HashMap<String, String> address = new HashMap<>();
        address.put("addressType", "card");
        address.put("line1", params.getString("address"));
        address.put("line2", "");
        address.put("city", params.getString("city"));
        address.put("state", params.getString("state"));
        address.put("country", "US");
        address.put("postalCode", params.getString("postalCode"));
        debitCard.put("address", address);
        data.put("debitCard", debitCard);
        vgsForm.setCustomData(data);

        // create request
        VGSRequest request = new VGSRequest.VGSRequestBuilder()
                .setMethod(HTTPMethod.PATCH)
                .setPath("/v1/contact/" + params.getString("contactId") + "/debitcard")
                .build();

        // call vgs collect link api
        vgsForm.asyncSubmit(request);
        vgsForm.addOnResponseListeners(new VgsCollectResponseListener() {
            @Override
            public void onResponse(VGSResponse response) {
                WritableMap result = new WritableNativeMap();
                try {
                    int code = response.getCode();
                    String body = response.getBody();
                    result.putInt("code", code);
                    if (code == 200) {
                        result.putString("response", body);
                        callback.invoke(result);
                        return;
                    }
                    WritableMap error = new WritableNativeMap();
                    error.putString("error_id", code + "_ERROR_TYPE");
                    error.putString("debug_message", body);
                    WritableMap debugRequest = new WritableNativeMap();
                    debugRequest.putString("path", request.getPath());
                    debugRequest.putString("method", request.getMethod().toString());
                    debugRequest.putString("format", request.getFormat().toString());
                    error.putMap("debug_request", debugRequest);
                    result.putMap("error", error);
                    callback.invoke(result);
                } catch (Throwable e) {
                    WritableMap error = new WritableNativeMap();
                    error.putString("error_id", "INTERNAL_ERROR_TYPE");
                    error.putString("debug_message", e.getMessage());
                    result.putMap("error", error);
                    callback.invoke(result);
                }
            }
            protected void onDestroy() {
                vgsForm.onDestroy();
            }
        });
    }
}