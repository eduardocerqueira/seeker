//date: 2023-05-30T16:40:28Z
//url: https://api.github.com/gists/1b4e340655fde4517ac769c270630146
//owner: https://api.github.com/users/asengsaragih

package com.suncode.relicbatik.ui.activity;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.viewpager2.widget.ViewPager2;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;

import com.google.android.material.tabs.TabLayout;
import com.google.android.material.tabs.TabLayoutMediator;
import com.suncode.relicbatik.R;
import com.suncode.relicbatik.adapter.IntroCategoryAdapter;
import com.suncode.relicbatik.base.BaseActivity;
import com.suncode.relicbatik.ui.fragment.intro.IntroFiveFragment;
import com.suncode.relicbatik.ui.fragment.intro.IntroFourFragment;
import com.suncode.relicbatik.ui.fragment.intro.IntroOneFragment;
import com.suncode.relicbatik.ui.fragment.intro.IntroThreeFragment;
import com.suncode.relicbatik.ui.fragment.intro.IntroTwoFragment;

public class IntroActivity extends BaseActivity implements View.OnClickListener {

    //tablayout and viewpager
    private TabLayout mTabLayout;
    private ViewPager2 mViewPager;
    private IntroCategoryAdapter mPagerAdapter;

    private TextView mSkipTextview;
    private TextView mDoneTextview;

    @Override
    protected void onStart() {
        super.onStart();
        if (session.getIntroStatus()) {
            startActivity(new Intent(IntroActivity.this, MainActivity.class));
            finish();
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_intro);

        init();
        setData();
    }

    private void init() {
        //viewpage and tab
        mTabLayout = findViewById(R.id.tabLayout_intro);
        mViewPager = findViewById(R.id.viewPager_intro);
        mPagerAdapter = new IntroCategoryAdapter(getSupportFragmentManager(), getLifecycle());

        //skip and done
        mSkipTextview = findViewById(R.id.textView_intro_skip);
        mDoneTextview = findViewById(R.id.textView_intro_done);
    }

    private void setData() {
        mSkipTextview.setOnClickListener(this);
        mDoneTextview.setOnClickListener(this);

        //setup view pager and tab layout
        mPagerAdapter.addFragment(new IntroOneFragment());
        mPagerAdapter.addFragment(new IntroTwoFragment());
        mPagerAdapter.addFragment(new IntroThreeFragment());
        mPagerAdapter.addFragment(new IntroFourFragment());
        mPagerAdapter.addFragment(new IntroFiveFragment());

        mViewPager.setAdapter(mPagerAdapter);
        mViewPager.registerOnPageChangeCallback(new ViewPager2.OnPageChangeCallback() {
            @Override
            public void onPageSelected(int position) {
                super.onPageSelected(position);

                //fungsi untuk menghilangkan dan menampilkan skip dan done button
                if (position == 4) {
                    mDoneTextview.setVisibility(View.VISIBLE);
                    mSkipTextview.setVisibility(View.GONE);
                } else {
                    mDoneTextview.setVisibility(View.GONE);
                    mSkipTextview.setVisibility(View.VISIBLE);
                }
            }
        });

        TabLayoutMediator mediator = new TabLayoutMediator(mTabLayout, mViewPager, (tab, position) -> {
        });
        mediator.attach();
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.textView_intro_skip:
                skip();
                break;
            case R.id.textView_intro_done:
                done();
                break;
        }
    }

    //function for button skip in intro
    private void skip() {
        AlertDialog.Builder builder = dialogMessage(IntroActivity.this, getString(R.string.dialog_skip_title), getString(R.string.dialog_skip_desc));

        builder.setPositiveButton(getString(R.string.skip), (dialog, which) -> {
            //save session
            session.setIntroStatus(true);

            startActivity(new Intent(getApplicationContext(), MainActivity.class));
            finish();
        });

        builder.setNegativeButton(getString(R.string.cancel), (dialog, which) -> dialog.dismiss());

        AlertDialog dialog = builder.create();
        dialog.show();
    }

    //function for button done in intro
    private void done() {
        //save session
        session.setIntroStatus(true);

        startActivity(new Intent(getApplicationContext(), MainActivity.class));
        finish();
    }
}