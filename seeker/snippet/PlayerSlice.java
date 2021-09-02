//date: 2021-09-02T16:57:10Z
//url: https://api.github.com/gists/f58b558cea4777769740b35a53803207
//owner: https://api.github.com/users/ZhangAnmy

    private void clickOnSoundOff() {
        textSeanceName.setVisibility(Component.HIDE);
        textDuration.setVisibility(Component.HIDE);
        loader.setVisibility(Component.HIDE);
//        buttonSoundOff.setVisibility(Component.VISIBLE);
        buttonSoundOff.setVisibility(Component.HIDE); // It's better hide it,please check
        buttonPause.setVisibility(Component.HIDE);
        buttonPlay.setVisibility(Component.HIDE);
        layoutSound.setVisibility(Component.VISIBLE);
        imageValidate.setVisibility(Component.VISIBLE);
        
        //Add below code
        soundProgressBar.setTouchFocusable(true);
        soundProgressBar.requestFocus();
        soundProgressBar.setRotationEventListener((component, rotationEvent) -> {
            if (rotationEvent != null) {
                //
                float rotationValue = rotationEvent.getRotationValue();
                if (Math.abs(rotationEventCount) == FACTOR) {
                    soundProgressBar.setProgressValue(soundProgressBar.getProgress() + rotationEventCount / FACTOR);
                    rotationEventCount = 0;
                } else {
                    rotationEventCount += rotationValue > 0 ? -1 : 1;
                }
                audioManager.changeVolumeBy(AudioManager.AudioVolumeType.STREAM_MUSIC,rotationEventCount);
                return true;
            }
            return false;
        });
    }

//....
    private void clickOnValdiateSound() {
        imageValidate.setVisibility(Component.HIDE);
        layoutSound.setVisibility(Component.HIDE);
        loader.setVisibility(Component.VISIBLE);
        buttonSoundOff.setVisibility(Component.VISIBLE);
        buttonPlay.setVisibility(Component.HIDE);
        buttonPause.setVisibility(Component.VISIBLE);
        textSeanceName.setVisibility(Component.VISIBLE);
        textDuration.setVisibility(Component.VISIBLE);
        //Add below 2 lines
        soundProgressBar.setTouchFocusable(false);
        soundProgressBar.clearFocus();
    }

