//date: 2023-05-18T17:02:30Z
//url: https://api.github.com/gists/ec15447e119405676320b5fa3be0a4d5
//owner: https://api.github.com/users/pranavpa8788

package com.example.androidtest;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.fragment.app.Fragment;

public class TextFrag extends Fragment {
    public static TextFrag new_instance(int position) {
        Bundle args = new Bundle();

        TextFrag fragment = new TextFrag();
        args.putInt("position", position);
        fragment.setArguments(args);
        return fragment;
    }


    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle state) {
        View result = inflater.inflate(R.layout.frag, container, false);
        TextView tv = result.findViewById(R.id.text_frag);
        int position = getArguments().getInt("position", -1);

        tv.setText("Retrieved position: " + String.valueOf(position));

        return result;
    }
}