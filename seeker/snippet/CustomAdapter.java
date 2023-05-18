//date: 2023-05-18T17:02:30Z
//url: https://api.github.com/gists/ec15447e119405676320b5fa3be0a4d5
//owner: https://api.github.com/users/pranavpa8788

package com.example.androidtest;

import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentActivity;
import androidx.viewpager2.adapter.FragmentStateAdapter;

public class CustomAdapter extends FragmentStateAdapter {
    public CustomAdapter(FragmentActivity fa) {
        super (fa);
    }

    @Override
    public Fragment createFragment(int position) {
        return TextFrag.new_instance(position);
    }

    @Override
    public int getItemCount() {
        return 3;
    }

}