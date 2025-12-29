//date: 2025-12-29T17:08:12Z
//url: https://api.github.com/gists/df225de3aafe60ae87c4cc2b53355417
//owner: https://api.github.com/users/sajjadyousefnia

package com.asantech.asanpay.user;

import android.annotation.SuppressLint;
import android.content.Context;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.android.volley.DefaultRetryPolicy;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import com.asantech.asanpay.setting.Globals;
import com.asantech.asanpay.R;
import com.google.android.material.bottomsheet.BottomSheetDialog;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

public class MessagesActivity extends AppCompatActivity {

    private RcyclerAdapter mrecyclerAdapter;
    private RequestQueue requestQueue;
    private StringRequest stringRequest;
    RecyclerView mrecyclerView;

    @SuppressLint("NotifyDataSetChanged")
    @Override
    public void onResume() {
        super.onResume();
        // داده‌های جدید را بارگذاری کنید
        Objects.requireNonNull(mrecyclerView.getAdapter()).notifyDataSetChanged();
    }

    @SuppressLint("NotifyDataSetChanged")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_messages);
        Objects.requireNonNull(getSupportActionBar()).hide();

        findViewById(R.id.back_ibtn).setOnClickListener(v -> finish());

        mrecyclerView = findViewById(R.id.recycler_view);
        mrecyclerView.setLayoutManager(new LinearLayoutManager(this, RecyclerView.VERTICAL, false));
        mrecyclerAdapter = new RcyclerAdapter(MessagesActivity.this, new JSONArray(), new JSONArray(), new JSONArray(), new JSONArray(), new JSONArray(), new JSONArray(), new JSONArray(), new JSONArray());
        mrecyclerAdapter.notifyDataSetChanged();
        mrecyclerView.setAdapter(mrecyclerAdapter);

        ApplyResponse();
        GetMessages();

    }

    private void GetMessages() {
        requestQueue = Volley.newRequestQueue(MessagesActivity.this);
        String req_url = Globals.global_link + "user_messages.php";
        stringRequest = new StringRequest(Request.Method.POST, req_url, response -> {
            try {
                JSONObject jresponse = new JSONObject(response);
                boolean status = jresponse.getBoolean("status");
                String message = jresponse.getString("message");
                if (status) {
                    if (Globals.getResponseMessages(MessagesActivity.this) == null) {
                        Globals.setResponseMessages(MessagesActivity.this, response);
                        ApplyResponse();
                    } else {
                        if (!Globals.getResponseMessages(MessagesActivity.this).equals(response)) {
                            Globals.setResponseMessages(MessagesActivity.this, response);
                            ApplyResponse();
                        }
                    }
                } else {
                    Toast.makeText(MessagesActivity.this, message.split(":")[1], Toast.LENGTH_LONG).show();
                }
            } catch (JSONException ignored) {
            }
        }, error -> {
            Toast.makeText(MessagesActivity.this, getString(R.string.severError), Toast.LENGTH_LONG).show();
        }) {
            @Override
            protected Map<String, String> getParams() {
                Map<String, String> params = new HashMap<>();
                params.put("private_key", Globals.getUser(MessagesActivity.this));
                return params;
            }
        };
        stringRequest.setRetryPolicy(new DefaultRetryPolicy(100000, 0, DefaultRetryPolicy.DEFAULT_BACKOFF_MULT));
        stringRequest.setTag("ALL");
        requestQueue.add(stringRequest);
    }

    private void SetMessagesSeen(String msgid) {
        requestQueue = Volley.newRequestQueue(MessagesActivity.this);
        String req_url = Globals.global_link + "user_seen.php";
        stringRequest = new StringRequest(Request.Method.POST, req_url, response -> {
            try {
                JSONObject jresponse = new JSONObject(response);
                boolean status = jresponse.getBoolean("status");
                String message = jresponse.getString("message");
                if (status) {
                    GetMessages();
                } else {
                    Toast.makeText(MessagesActivity.this, message.split(":")[1], Toast.LENGTH_LONG).show();
                }
            } catch (JSONException ignored) {
            }
        }, error -> {
            Toast.makeText(MessagesActivity.this, getString(R.string.severError), Toast.LENGTH_LONG).show();
        }) {
            @Override
            protected Map<String, String> getParams() {
                Map<String, String> params = new HashMap<>();
                params.put("private_key", Globals.getUser(MessagesActivity.this));
                params.put("msg_id", msgid);
                return params;
            }
        };
        stringRequest.setRetryPolicy(new DefaultRetryPolicy(100000, 0, DefaultRetryPolicy.DEFAULT_BACKOFF_MULT));
        stringRequest.setTag("ALL");
        requestQueue.add(stringRequest);
    }

    private void ApplyResponse() {
        if (Globals.getResponseMessages(MessagesActivity.this) != null) {
            try {
                JSONObject jresponse = new JSONObject(Globals.getResponseMessages(MessagesActivity.this));
                boolean status = jresponse.getBoolean("status");
                JSONArray msgs_id = jresponse.getJSONArray("msgs_id");
                JSONArray msgs_sender = jresponse.getJSONArray("msgs_sender");
                JSONArray msgs_receiver = jresponse.getJSONArray("msgs_receiver");
                JSONArray msgs_title = jresponse.getJSONArray("msgs_title");
                JSONArray msgs_message = jresponse.getJSONArray("msgs_message");
                JSONArray msgs_product = jresponse.getJSONArray("msgs_product");
                JSONArray msgs_seen = jresponse.getJSONArray("msgs_seen");
                JSONArray msgs_time = jresponse.getJSONArray("msgs_time");
                if (status) {
                    View empty = findViewById(R.id.empty);
                    if (msgs_id.length() == 0) {
                        empty.setVisibility(View.VISIBLE);
                    } else {
                        empty.setVisibility(View.GONE);
                    }
                    mrecyclerAdapter.Update(MessagesActivity.this, msgs_id, msgs_sender, msgs_receiver, msgs_title, msgs_message, msgs_product, msgs_seen, msgs_time);
                }
            } catch (JSONException ignored) {
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (requestQueue != null) {
            requestQueue.cancelAll("ALL");
        }
        if (stringRequest != null) {
            stringRequest.cancel();
        }
    }

    private class RcyclerAdapter extends RecyclerView.Adapter<RcyclerAdapter.mViewHolder> {

        Context context;
        JSONArray msgs_id, msgs_sender, msgs_receiver, msgs_title, msgs_message, msgs_product, msgs_seen, msgs_time;

        private RcyclerAdapter(Context context, JSONArray msgs_id, JSONArray msgs_sender, JSONArray msgs_receiver, JSONArray msgs_title, JSONArray msgs_message, JSONArray msgs_product, JSONArray msgs_seen, JSONArray msgs_time) {
            this.context = context;
            this.msgs_id = msgs_id;
            this.msgs_sender = msgs_sender;
            this.msgs_receiver = msgs_receiver;
            this.msgs_title = msgs_title;
            this.msgs_message = msgs_message;
            this.msgs_product = msgs_product;
            this.msgs_seen = msgs_seen;
            this.msgs_time = msgs_time;
        }

        public void Update(Context context, JSONArray msgs_id, JSONArray msgs_sender, JSONArray msgs_receiver, JSONArray msgs_title, JSONArray msgs_message, JSONArray msgs_product, JSONArray msgs_seen, JSONArray msgs_time) {
            this.context = context;
            this.msgs_id = msgs_id;
            this.msgs_sender = msgs_sender;
            this.msgs_receiver = msgs_receiver;
            this.msgs_title = msgs_title;
            this.msgs_message = msgs_message;
            this.msgs_product = msgs_product;
            this.msgs_seen = msgs_seen;
            this.msgs_time = msgs_time;
            notifyDataSetChanged();
        }

        @NonNull
        @Override
        public RcyclerAdapter.mViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            View itemView = LayoutInflater.from(parent.getContext()).inflate(R.layout.messages_list_item, parent, false);
            return new RcyclerAdapter.mViewHolder(itemView);
        }

        @Override
        public void onBindViewHolder(@NonNull mViewHolder holder, int position) {
            try {
                holder.titile_tv.setText(msgs_title.getString(position));
                holder.msg_tv.setText(msgs_message.getString(position));
                holder.time_tv.setText(msgs_time.getString(position));
                if (msgs_seen.getInt(position) == 0) {
                    holder.view_iv.setColorFilter(context.getResources().getColor(R.color.colorPrimary));
                } else {
                    holder.view_iv.setColorFilter(context.getResources().getColor(R.color.gray_light_color));
                }
                holder.itemView.setOnClickListener(view -> {
                    try {
                        SetMessagesSeen(msgs_id.getString(position));
                    } catch (JSONException ignored) {
                    }
                    View view_sheet = getLayoutInflater().inflate(R.layout.bottom_sheet_message, null);
                    BottomSheetDialog dialog = new BottomSheetDialog(context);
                    view_sheet.findViewById(R.id.cancel_btn).setOnClickListener(view1 -> dialog.dismiss());
                    TextView title_tv = view_sheet.findViewById(R.id.title_tv);
                    TextView message_tv = view_sheet.findViewById(R.id.message_tv);
                    TextView time_tv = view_sheet.findViewById(R.id.time_tv);
                    try {
                        title_tv.setText(msgs_title.getString(position));
                        message_tv.setText(msgs_message.getString(position));
                        time_tv.setText(msgs_time.getString(position));
                    } catch (JSONException ignored) {
                    }
                    dialog.setContentView(view_sheet);
                    dialog.show();
                });
            } catch (JSONException ignored) {
            }
        }

        @Override
        public int getItemCount() {
            return msgs_id.length();
        }

        class mViewHolder extends RecyclerView.ViewHolder {

            TextView titile_tv, msg_tv, time_tv;
            ImageView view_iv;

            mViewHolder(@NonNull View itemView) {
                super(itemView);
                titile_tv = itemView.findViewById(R.id.titile_tv);
                msg_tv = itemView.findViewById(R.id.msg_tv);
                time_tv = itemView.findViewById(R.id.time_tv);
                view_iv = itemView.findViewById(R.id.view_iv);
            }
        }
    }
}