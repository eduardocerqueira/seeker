//date: 2025-07-04T16:33:18Z
//url: https://api.github.com/gists/9b3051035ffa8a4f83a0afee91227ed5
//owner: https://api.github.com/users/SGlowQ

package com.appshare.android.ilisten.watch.ui.home;

import android.annotation.SuppressLint;
import android.os.CountDownTimer;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;
import java.util.Calendar;

public class HomeViewModel extends ViewModel {

    private final MutableLiveData<String> mText;
    private final MutableLiveData<Integer> mTextColor;

    public HomeViewModel() {
        mText = new MutableLiveData<>();
        mTextColor = new MutableLiveData<>();

        // 设置高考时间为2026年6月7日00:00:00
        Calendar calendar = Calendar.getInstance();
        calendar.set(2026, Calendar.JUNE, 7, 0, 0, 0);
        long endTime = calendar.getTimeInMillis();
        long currentTime = System.currentTimeMillis();
        long remainingTime = endTime - currentTime;

        if (remainingTime <= 0) {
            mText.setValue("高考已结束");
            mTextColor.setValue(android.graphics.Color.RED); // 红色
        } else {
            // 倒计时逻辑（每秒更新一次）
            new CountDownTimer(remainingTime, 1000) {
                @Override
                public void onTick(long millisUntilFinished) {
                    // 计算剩余时间（天、时、分、秒）
                    long days = millisUntilFinished / (1000 * 60 * 60 * 24);
                    long hours = (millisUntilFinished % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60);
                    long minutes = (millisUntilFinished % (1000 * 60 * 60)) / (1000 * 60);
                    long seconds = (millisUntilFinished % (1000 * 60)) / 1000;

                    // 格式化倒计时文本
                    @SuppressLint("DefaultLocale")
                    String countdownText = String.format("2026\n高考倒计时：%d天 %02d:%02d:%02d",
                            days, hours, minutes, seconds);
                    mText.setValue("\n" + countdownText);

                    // 根据剩余天数动态更新文本颜色
                    if (days > 90) {
                        mTextColor.setValue(android.graphics.Color.GREEN); // 绿色
                    } else if (days > 60) {
                        mTextColor.setValue(0xFF9ACD32); // 黄绿色
                    } else if (days > 30) {
                        mTextColor.setValue(android.graphics.Color.YELLOW); // 黄色
                    } else if (days > 7) {
                        mTextColor.setValue(0xFFFFA500); // 橙色
                    } else {
                        mTextColor.setValue(android.graphics.Color.RED); // 红色
                    }
                }

                @Override
                public void onFinish() {
                    mText.setValue("\n高考时间到！");
                    mTextColor.setValue(android.graphics.Color.RED); // 红色
                }
            }.start();
        }
    }

    public LiveData<String> getText() {
        return mText;
    }

    public LiveData<Integer> getTextColor() {
        return mTextColor;
    }
}