//date: 2024-08-29T17:03:49Z
//url: https://api.github.com/gists/a52fb5d70dcef6d431363b0f431de4e6
//owner: https://api.github.com/users/pagetronic

package live.page.wiki;

import info.bliki.wiki.filter.PlainTextConverter;
import info.bliki.wiki.model.WikiModel;
import live.page.hubd.system.json.Json;
import live.page.hubd.system.utils.Fx;
import org.apache.commons.text.StringEscapeUtils;

import java.io.IOException;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class WikiParser extends WikiModel {

    final Json data = new Json();

    public WikiParser() {
        super("", "");
    }

    public static Json getInfos(String title, String text) {
        WikiParser wikiModel = new WikiParser();
        try {
            wikiModel.render(new PlainTextConverter(), text, new StringBuilder(), true, true);
        } catch (IOException e) {
            Fx.log("\n#{" + url(title) + "}");
        }

        Json data = new Json();

        if (wikiModel.data.containsKey("information")) {

            Json information = wikiModel.data.getJson("information");

            Json description = new Json();
            if (information.containsKey("description") && String.class.isAssignableFrom(information.get("description").getClass())) {
                description.put("int", information.getString("description"));
            } else {
                description = information.getJson("description");
                if (description != null && description.containsKey("langswitch")) {
                    description = description.getJson("langswitch");
                }
            }
            data.put("description", description);

            if (description == null || description.isEmpty()) {
                Fx.log("\nD{" + url(title) + "}");
                data.put("description", StringEscapeUtils.unescapeHtml4(title).split("\\.")[0]);
            }
            if (information.containsKey("author")) {
                data.put("author", information.get("author"));
            }

        }
        List<Double> coordinates = WikiLocation.findLocation(wikiModel.data);
        if (coordinates == null) {
            Fx.log("\nL{" + url(title) + "}");
        }
        data.put("coordinates", coordinates);
        data.put("data", wikiModel.data);
        return data;
    }

    private static String url(String title) {
        return " https://commons.wikimedia.org/wiki/" + URLEncoder.encode(StringEscapeUtils.unescapeHtml4(title), StandardCharsets.UTF_8).replace("+", "%20") + " ";
    }


    @Override
    public void substituteTemplateCall(String templateName, Map<String, String> parameterMap, Appendable writer) throws IOException {

        writer.append("@@Template@").append(templateName.toLowerCase().trim()).append("@");

        Json params = new Json();

        for (String key : parameterMap.keySet()) {

            WikiParser model = new WikiParser();
            StringBuilder builder = new StringBuilder();
            model.render(new PlainTextConverter(), parameterMap.get(key), builder, true, false);
            String str = builder.toString().replace("[\r\n ]+", " ").replaceAll(" +", " ").trim();
            writer.append(str);
            Matcher match = Pattern.compile("@@Template@([^@]+)@", Pattern.MULTILINE).matcher(builder.toString());
            Json done = new Json();
            while (match.find()) {
                if (data.containsKey(match.group(1).toLowerCase().trim())) {
                    done.put(match.group(1).toLowerCase().trim(), data.get(match.group(1).toLowerCase().trim()));
                    data.remove(match.group(1).toLowerCase().trim());
                }
            }
            if (!key.equals("prec") && !key.equals("wikidata")) {
                if (!done.isEmpty()) {
                    params.put(key.toLowerCase().trim(), done);
                } else if (!str.isEmpty()) {
                    params.put(key.toLowerCase().trim(), str);
                }
            }
        }

        if (params.size() == 1 && params.containsKey("1")) {
            data.put(templateName.toLowerCase().trim(), params.get("1"));
        } else if (params.keySet().stream().allMatch(name -> name.matches("[0-9]+"))) {
            data.put(templateName.toLowerCase().trim(), params.values().stream().toList());
        } else {
            data.put(templateName.toLowerCase().trim(), params);
        }
    }

}