//date: 2025-09-30T16:56:55Z
//url: https://api.github.com/gists/0b93b9d1f02c6cf903a69160980a259b
//owner: https://api.github.com/users/Starfruit2210

public static void runSchemaFile(DataSource dataSource, String resourcePath) throws Exception {
        try (InputStream input = MineJob.class.getResourceAsStream(resourcePath)) {
            if (input == null) {
                throw new IllegalArgumentException("Schema resource not found: " + resourcePath);
            }

            try (BufferedReader reader = new BufferedReader(new InputStreamReader(input, StandardCharsets.UTF_8))) {
                Connection connection = dataSource.getConnection();
                Statement statement = connection.createStatement();

                StringBuilder queryBuilder = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.trim().startsWith("--")) continue;

                    queryBuilder.append(line).append('\n');
                    if (line.trim().endsWith(";")) {
                        String query = queryBuilder.toString().trim();
                        if (!query.isEmpty()) {
                            statement.execute(query.substring(0, query.length() - 1));
                        }
                        queryBuilder.setLength(0);
                    }
                }
                if (!queryBuilder.isEmpty()) {
                    statement.execute(queryBuilder.toString());
                }
            }
        }
    }