//date: 2025-02-07T17:09:53Z
//url: https://api.github.com/gists/7196a5e0737ee7def82442340824a222
//owner: https://api.github.com/users/mihaisdm

class Scratch {
    public static void main(String[] args) {


        long v1 = parseVersion("1.2.3-4.5.6-strinf");
        long v2 = parseVersion("1.2.3-4.6.0-beta");
        long v3 = parseVersion("1.3-4.6.0-beta");
        long v4 = parseVersion("1.2.3-4.6.0-beta");
        long v5 = parseVersion("1.2.3-4.6.0-beta");
        long v6 = parseVersion("1.2.3-4.6.0-beta");
        long v7 = parseVersion("1.2.3-4.6.0-beta");
        long v8 = parseVersion("1.2.3-4.6.0-beta");

//        long v = parseVersion("1");
        assert parseVersion("1") > System.currentTimeMillis();
        assert parseVersion("0") > System.currentTimeMillis();


        assert v1 > v2;
        assert v3 > v2 && v3 > v1;
    }

    /**
     * Parses a version string of the form "1.2.3-4.5.6-qualifier"
     * and packs the first six numeric segments (major, minor, patch,
     * buildMajor, buildMinor, buildPatch) into a long value.
     *
     * <p>The numeric parts are assumed to be less than 1024 (i.e. they fit
     * in 10 bits). Missing parts are treated as 0. The qualifier (if any)
     * is ignored.</p>
     *
     * @param version the version string to parse
     * @return a long value encoding the version
     */
    public static long parseVersion(String version) {
        // We expect up to 6 numeric segments.
        int[] segments = new int[6];
        int index = 0;

        // First split the version on '-' characters.
        // For example, "1.2.3-4.5.6-strinf" splits into:
        //    [ "1.2.3", "4.5.6", "strinf" ]
        // We ignore non-numeric parts.
        String[] dashParts = version.split("-");
        for (String part : dashParts) {
            // Split further by '.' to get individual numbers.
            String[] dotParts = part.split("\\.");
            for (String s : dotParts) {
                if (index < segments.length) {
                    try {
                        segments[index] = Integer.parseInt(s);
                    } catch (NumberFormatException e) {
                        // If this part is not a number, ignore it.
                    }
                    index++;
                }
            }
        }

        // Now, combine the six segments into one long.
        // We allocate 10 bits for each segment.
        // The most significant bits come from the first segment.
        long result = 0;
        for (int seg : segments) {
            result = (result << 10) | (seg & 0x3FF);
        }
        return result;
    }
}