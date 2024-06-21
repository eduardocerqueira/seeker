//date: 2024-06-21T17:02:15Z
//url: https://api.github.com/gists/e0eea7f3f78d1a59447f5225761426f7
//owner: https://api.github.com/users/dlabey

package com.sony.sie.payments.paas.apigateway;

import java.util.regex.Pattern;

import org.apache.commons.lang3.StringUtils;

public final class Utils {

    private static final String PATH_SEPARATOR = "/";

    private Utils() {
        throw new IllegalStateException("Utility class");
    }

    public static String getPathParameter(String pathTemplate, String path, String paramTemplate) {
        String pathTemplateDelimiter = Pattern.quote(paramTemplate);
        String[] pathTemplatePieces = pathTemplate.split(pathTemplateDelimiter);
        int pathTemplateSeparatorPosition = StringUtils.countMatches(pathTemplatePieces[0], PATH_SEPARATOR);
        String[] pathPieces = path.split(PATH_SEPARATOR);
        String param = null;
        if (pathTemplateSeparatorPosition <= pathPieces.length) {
            param = pathPieces[pathTemplateSeparatorPosition];
        }

        return param;
    }
}
