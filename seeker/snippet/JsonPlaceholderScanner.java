//date: 2025-03-24T16:57:07Z
//url: https://api.github.com/gists/9fab7639c5ae96c1bf4617b89f704138
//owner: https://api.github.com/users/jyemin

/*
 * Copyright 2008-present MongoDB, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.bson.json;

import static java.lang.System.out;

class JsonPlaceholderScanner {

    public static String replace(String json) {
        StringBuilder builder = new StringBuilder(json.length());

        int i = 0;
        while (i < json.length()) {
            char c = json.charAt(i++);
            switch (c) {
                case '{':
                case '}':
                case '[':
                case ']':
                case ':':
                case ',':
                case ' ':
                    builder.append(c);
                    break;
                case '\'':
                case '"':
                    i = scanString(c, i, json, builder);
                    break;
                case '?':
                    builder.append("{$undefined: true}");
                    break;
                default:
                    if (c == '-' || Character.isDigit(c)) {
                        i = scanNumber(c, i, json, builder);
                    } else if (c == '$' || c == '_' || Character.isLetter(c)) {
                        i = scanUnquotedString(c, i, json, builder);
                    } else {
                        builder.append(c);  // or throw exception, as this isn't valid JSON
                    }
            }
        }
        return builder.toString();
    }

    private static int scanNumber(char firstCharacter, int startIndex, String json, StringBuilder builder) {
        builder.append(firstCharacter);
        int i = startIndex;
        char c = json.charAt(i++);
        while (i < json.length() && Character.isDigit(c)) {
            builder.append(c);
            c = json.charAt(i++);
        }
        return i - 1;
    }

    private static int scanUnquotedString(final char firstCharacter, final int startIndex, final String json, final StringBuilder builder) {
        builder.append(firstCharacter);
        int i = startIndex;
        char c = json.charAt(i++);
        while (i < json.length() && Character.isLetterOrDigit(c)) {
            builder.append(c);
            c = json.charAt(i++);
        }
        return i - 1;
    }

    private static int scanString(final char quoteCharacter, final int startIndex, final String json, final StringBuilder builder) {
        int i = startIndex;
        builder.append(quoteCharacter);
        while (i < json.length()) {
            char c = json.charAt(i++);
            if (c == quoteCharacter) {
                builder.append(c);
                return i;
            } else {
                builder.append(c);
            }
        }
        return i;
    }

    public static void main(String[] args) {
        out.println(replace("{}"));
        out.println(replace("{x: 1}"));
        out.println(replace("{x: -12345}"));
        out.println(replace("{x: 1, y: \"abc\"}"));
        out.println(replace("{x: 1, y: 'abc'}"));
        out.println(replace("{_id: 1, y: 'abc'}"));
        out.println(replace("{x: 1, y: {$numberLong: \"12345\"}}"));
        out.println(replace("{x: ?, y: ?}"));
        out.println(replace("{x: ?1, y: ?}"));
        out.println(replace("{x: 1, y: '?'}"));
        out.println(replace("{_?: 1, y: '?'}"));
    }
}
