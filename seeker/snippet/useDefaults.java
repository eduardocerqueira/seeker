//date: 2025-05-15T16:48:46Z
//url: https://api.github.com/gists/4d5a3572c86bf1033e10c7714fbb92e7
//owner: https://api.github.com/users/harrel56

    @Test
    void q() {
        String schema = """
                {
                            "$schema": "https://json-schema.org/draft/2020-12/schema",
                            "type": "object",
                            "properties": {
                              "a": {
                                "type": "string",
                                "default": "is a"
                              },
                              "b": {
                                "type": "string",
                                "default": "is b"
                              }
                            },
                            "useDefaults": true
                        }""";

        String inst = """
                {
                  "a": "foo"
                }""";

        EvaluatorFactory factory = new EvaluatorFactory.Builder().withKeyword("useDefaults", (ctx, node) -> {
            Map<String, JsonNode> defaultsMap = new HashMap<>();
            // todo: add null checks
            for (Map.Entry<String, JsonNode> entry : ctx.getCurrentSchemaObject().get("properties").asObject().entrySet()) {
                defaultsMap.put(entry.getKey(), entry.getValue().asObject().get("default"));
            }
            return (a, b) -> Evaluator.Result.success(defaultsMap);
        }).build();
        Validator.Result res = new ValidatorFactory().withEvaluatorFactory(factory).validate(schema, inst);
    }