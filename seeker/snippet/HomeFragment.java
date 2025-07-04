//date: 2025-07-04T16:33:00Z
//url: https://api.github.com/gists/9175f53b8112edacbe84c361a9107545
//owner: https://api.github.com/users/SGlowQ

package com.appshare.android.ilisten.watch.ui.home;

import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;

import com.appshare.android.ilisten.watch.databinding.FragmentHomeBinding;

public class HomeFragment extends Fragment {

    private static final String TAG = "HomeFragment";
    private FragmentHomeBinding binding;
    private TextView countdownTextView;

    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {
        HomeViewModel homeViewModel =
                new ViewModelProvider(this).get(HomeViewModel.class);

        binding = FragmentHomeBinding.inflate(inflater, container, false);
        View root = binding.getRoot();

        // 初始化倒计时 TextView
        countdownTextView = binding.textHome;

        // 观察文本内容变化
        homeViewModel.getText().observe(getViewLifecycleOwner(), this::updateCountdownText);

        // 观察文本颜色变化
        homeViewModel.getTextColor().observe(getViewLifecycleOwner(), this::updateTextColor);

        return root;
    }

    /**
     * 更新倒计时文本内容
     * @param text 新的文本内容
     */
    private void updateCountdownText(String text) {
        if (countdownTextView != null && text != null) {
            countdownTextView.setText(text);
            Log.d(TAG, "Countdown text updated: " + text);
        }
    }

    /**
     * 更新倒计时文本颜色
     * @param textColor 新的文本颜色
     */
    private void updateTextColor(@Nullable Integer textColor) {
        if (countdownTextView != null && textColor != null) {
            countdownTextView.setTextColor(textColor);
            Log.d(TAG, "Text color updated: " + String.format("#%06X", (0xFFFFFF & textColor)));
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
        countdownTextView = null;
    }
}