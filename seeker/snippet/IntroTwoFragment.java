//date: 2023-05-30T16:40:28Z
//url: https://api.github.com/gists/1b4e340655fde4517ac769c270630146
//owner: https://api.github.com/users/asengsaragih

package com.suncode.relicbatik.ui.fragment.intro;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

import com.bumptech.glide.Glide;
import com.suncode.relicbatik.R;

public class IntroTwoFragment extends Fragment {

    public IntroTwoFragment() {
        // Required empty public constructor
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_intro, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        TextView title = getView().findViewById(R.id.textview_intro_title);
        ImageView icon = getView().findViewById(R.id.imageView_intro_image);
        TextView content = getView().findViewById(R.id.textview_intro_description);

        //set content
        title.setText(getContext().getString(R.string.intro_identification_title));
        Glide.with(getContext())
                .load(getContext().getDrawable(R.drawable.ic_intro_identification))
                .into(icon);

        content.setText(getContext().getString(R.string.intro_identification_description));
    }
}