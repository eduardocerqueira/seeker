//date: 2023-09-12T16:54:02Z
//url: https://api.github.com/gists/e5bf10fdf1b31e26b8d93bd4c09cade6
//owner: https://api.github.com/users/RakhimBek

public class Main { 
  private static final Pattern PROPERTY_PATTERN = Pattern.compile("\\{\\{\\s+\\.Values\\.(?<prop>[^}]+)}}", CASE_INSENSITIVE);

	public static void main(String[] args) throws IOException {
		// properties extractor
		//final var text = "<path-to-helm-templates>";
		final var file = new File(text);
		final var properties = new ArrayList<String>();
		try (
				final var inputStream = new FileInputStream(file);
				final var inputStreamReader = new InputStreamReader(inputStream);
				final var bufferedReader = new BufferedReader(inputStreamReader)
		) {
			String line;
			while ((line = bufferedReader.readLine()) != null) {
				if (line.contains("{{")) {
					properties.addAll(properties(line));
				}
			}
		}

		properties.stream().sorted().distinct().forEach(System.out::println);
	}

	private static List<String> properties(String line) {
		final var matcher = PROPERTY_PATTERN.matcher(line);
		final var properties = new ArrayList<String>();
		while (matcher.find()) {
			final var group = matcher.group("prop").trim();
			System.out.println(group);
			properties.add(group);
		}

		return properties;
	}
}