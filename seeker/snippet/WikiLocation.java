//date: 2024-08-29T17:03:49Z
//url: https://api.github.com/gists/a52fb5d70dcef6d431363b0f431de4e6
//owner: https://api.github.com/users/pagetronic

package live.page.wiki;

import live.page.hubd.system.json.Json;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class WikiLocation {
    public static List<Double> findLocation(Json data) {
        for (String key : data.keySet()) {
            if (key.matches("object location|object location dec|location|location dec|camera location|camera location dec")) {
                List<Double> coordinates = convertCoordinates(data.getList(key));
                if (coordinates != null) {
                    return coordinates;
                }
            }
            if (Json.class.isAssignableFrom(data.get(key).getClass())) {
                List<Double> coordinates = findLocation(data.getJson(key));
                if (coordinates != null) {
                    return coordinates;
                }
            }
            if (List.class.isAssignableFrom(data.get(key).getClass()) &&
                    !data.getList(key).isEmpty()) {
                for (Object item : data.getList(key)) {
                    if (item != null && Json.class.isAssignableFrom(item.getClass())) {
                        List<Double> coordinates = findLocation((Json) item);
                        if (coordinates != null) {
                            return coordinates;
                        }
                    }
                }
            }
        }
        return null;
    }


    private static List<Double> convertCoordinates(List<String> coordinates) {
        if (coordinates == null) {
            return null;
        }
        coordinates = new ArrayList<>(coordinates);
        for (int key : new int[]{8, 2}) {
            if (coordinates.size() > key) {
                for (String start : new String[]{
                        "source", "alt", "type",
                        "heading", "region", "zoom", "scale",
                        "...", "sl", "dim", "view"}) {
                    if (coordinates.get(key).trim().toLowerCase().startsWith(start) || coordinates.get(key).trim().isEmpty()) {
                        coordinates.remove(key);
                        break;
                    }
                }
            }
        }

        try {
            if (coordinates.size() >= 8 &&
                    (coordinates.get(3).equals("N") || coordinates.get(3).equals("S")) &&
                    (coordinates.get(7).equals("E") || coordinates.get(7).equals("W"))
            ) {
                return Arrays.asList(
                        convertCoordinates(Double.parseDouble(coordinates.get(0)), Double.parseDouble(coordinates.get(1)), Double.parseDouble(coordinates.get(2)), coordinates.get(3)),
                        convertCoordinates(Double.parseDouble(coordinates.get(4)), Double.parseDouble(coordinates.get(5)), Double.parseDouble(coordinates.get(6)), coordinates.get(7)));
            } else {
                return Arrays.asList(Double.parseDouble(coordinates.get(0)), Double.parseDouble(coordinates.get(1)));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    private static double convertCoordinates(double degree, double minute, double second, String heading) {
        double decimalDegrees = degree + (minute / 60.0) + (second / 3600.0);
        if ("W".equals(heading) || "S".equals(heading)) {
            decimalDegrees = -decimalDegrees;
        }

        return decimalDegrees;
    }
}
