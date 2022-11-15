//date: 2022-11-15T17:08:47Z
//url: https://api.github.com/gists/80e11f151293c5617a64f28494d14deb
//owner: https://api.github.com/users/AlexKolpa

import lombok.extern.slf4j.Slf4j;

@Slf4j
public class ColorTest {
	private static final long PRIME = 2305843009213693951L;
	private static final char[] ALPHABET = "abcdefghijklmnopqrstuvwxyz".toCharArray();

	public static void main(String[] args) {
		StringBuilder builder = new StringBuilder();
		for (char first: ALPHABET) {
			for (char second : ALPHABET) {
				String initials = first + "" + second;
				builder.append("console.log('%c")
						.append(initials)
						.append("', 'background: ")
						.append(getRgbHex(initials))
						.append("; color: white');")
						.append("\n");
			}
		}
		log.info("Result: {}", builder);
	}

	private static String getRgbHex(String initials) {
		long val = PRIME % initials.hashCode();

		int hue = normalize(val >> 1, 0, 360);
		int saturation = normalize(val >> 2, 0, 51);
		int lightness = normalize(val >> 4, 0, 62);

		return HSLtoRGB(hue, saturation, lightness);
	}

	private static int normalize(long val, int min, int max) {
		return (int) ((val % (max - min)) + min);
	}

	public static String HSLtoRGB(float h, float s, float l) {
		h = h % 360.0f;
		h /= 360f;
		s /= 100f;
		l /= 100f;

		float q;

		if (l < 0.5)
			q = l * (1 + s);
		else
			q = (l + s) - (s * l);

		float p = 2 * l - q;

		int r = Math.round(Math.max(0, hueToRGB(p, q, h + (1.0f / 3.0f)) * 256));
		int g = Math.round(Math.max(0, hueToRGB(p, q, h) * 256));
		int b = Math.round(Math.max(0, hueToRGB(p, q, h - (1.0f / 3.0f)) * 256));

		return String.format("#%06x", (r << 16) + (g << 8) + b);
	}

	private static float hueToRGB(float p, float q, float h) {
		if (h < 0)
			h += 1;

		if (h > 1)
			h -= 1;

		if (6 * h < 1) {
			return p + ((q - p) * 6 * h);
		}

		if (2 * h < 1) {
			return q;
		}

		if (3 * h < 2) {
			return p + ((q - p) * 6 * ((2.0f / 3.0f) - h));
		}

		return p;
	}
}