//date: 2022-09-27T17:14:04Z
//url: https://api.github.com/gists/d119ce3f4a86dd1d406f2d717de30429
//owner: https://api.github.com/users/IbremMiner837

package com.mcbedrock.minecraftnews.utils;

import android.content.Context;
import android.text.Html;
import android.util.Log;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.material.button.MaterialButton;
import com.google.mlkit.common.model.DownloadConditions;
import com.google.mlkit.common.model.RemoteModelManager;
import com.google.mlkit.nl.translate.TranslateLanguage;
import com.google.mlkit.nl.translate.TranslateRemoteModel;
import com.google.mlkit.nl.translate.Translation;
import com.google.mlkit.nl.translate.Translator;
import com.google.mlkit.nl.translate.TranslatorOptions;
import com.mcbedrock.minecraftnews.R;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class TranslationHelper {

    private static Context context;

    public TranslationHelper(Context context) {
        this.context = context;
    }

    private static final String TAG = "ArticleTranslationHelper";
    public static Boolean isTranslated = false;

    public static void translateArticle(String article, TextView textView, MaterialButton button) {

        button.setText(R.string.translating);
        translator().translate(article)
                .addOnSuccessListener(
                        translatedText -> {
                            // Translation successful.
                            Log.d(TAG, "onSuccess: Translation successful.");
                            textView.setText("");
                            textView.append(Html.fromHtml(((String) translatedText)));
                            button.setText("Перевести обратно");
                            isTranslated = true;
                        })
                .addOnFailureListener(
                        e -> {
                            // Error.
                            // ...
                            Log.d(TAG, "onFailure: Error.");
                            isTranslated = false;
                        });
    }

    public static boolean isLanguageDownloaded(String language) {
        boolean isDownloaded = false;
        for (int i = 0; i < getAvailableModels().size(); i++) {
            if (getAvailableModels().get(i).equals(language)) {
                isDownloaded = true;
            } else {
                isDownloaded = false;
            }
        }
        return isDownloaded;
    }

    public static List<String> getAvailableModels() {
        List<String> availableModels = new ArrayList<>();
        getRemoteModelManager()
                .getDownloadedModels(TranslateRemoteModel.class)
                .addOnSuccessListener(
                        models -> {
                            // Model downloading is complete.
                            // ...
                            for (TranslateRemoteModel model : models) {
                                availableModels.add(model.getLanguage());
                            }
                        })
                .addOnFailureListener(
                        e -> {
                            // Model downloading failed.
                            // ...
                            Log.d(TAG, "onFailure: Model downloading failed.");
                        });
        return availableModels;
    }

    public static void deleteModelTranslateRemoteModel() {
        getRemoteModelManager()
                .deleteDownloadedModel(getModel(getSystemLanguage()))
                .addOnSuccessListener(o -> {
                    new DialogsUtil().deletingTranslationModelDone(context);
                })
                .addOnFailureListener(e -> {
                    // Error.
                });
    }

    public static void downloadModel() {
        getRemoteModelManager()
                .download(getModel(getSystemLanguage()), setDownloadConditions())
                .addOnSuccessListener(o -> {
                    new DialogsUtil().translateModelDownloaded(context);
                })
                .addOnFailureListener(e -> {
                    Toast.makeText(context, "Error", Toast.LENGTH_SHORT).show();
                });
    }

    public static Translator translator() {
        return Translation.getClient(setOptions());
    }

    private static TranslatorOptions setOptions() {
        return new TranslatorOptions.Builder()
                .setSourceLanguage(TranslateLanguage.ENGLISH)
                .setTargetLanguage(getSystemLanguage())
                .build();
    }

    private static DownloadConditions setDownloadConditionsWithWifi() {
        return new DownloadConditions.Builder()
                .requireWifi()
                .build();
    }

    private static DownloadConditions setDownloadConditions() {
        return new DownloadConditions.Builder()
                .build();
    }

    private static RemoteModelManager getRemoteModelManager () {
        return RemoteModelManager.getInstance();
    }

    private static TranslateRemoteModel getModel(String languageCode) {
        return new TranslateRemoteModel.Builder(languageCode).build();
    }

    public static Boolean isTranslated() {
        return isTranslated;
    }

    public static String getSystemLanguage() {
        String language = Locale.getDefault().getLanguage();
        Log.d(TAG, "getSystemLanguage: " + language);
        return language;
    }
}
