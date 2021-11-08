//date: 2021-11-08T16:58:40Z
//url: https://api.github.com/gists/60443d9abc251df9baccd2263affe5f2
//owner: https://api.github.com/users/daoudfares

// KStream flow creation from "sales" topic
KStream<String, ProductRaw> sales = builder.stream(Serdes.String(), productRawSerde, "sales");

// KTable (table) creation from the "repository" topic
KTable<String, Repository> repository = builder.table(Serdes.String(), repositorySerde, "repository");
KStream<String, ProductEnriched> enriched = sales
	// Repartition of the flow with the new key that allow us to make a join
    .map((k, v) -> new KeyValue<>(v.getId().toString(), v))
 	// Copying the flow to a new topic with the new key 
    .through(Serdes.String(), productRawSerde, "sales-by-product-id")
    // Join the sales flow with the repository
    .leftJoin(repository, (sale, ref) -> {
        if (ref == null) return new ProductEnriched(sale.getId(), "UNKNOWN ID", sale.getPrice());
        else return new ProductEnriched(sale.getId(), ref.getName(), sale.getPrice());
    });

// Publish the flow to the "sales-enriched" topic
enriched.to(Serdes.String(), productEnrichedSerde, "sales-enriched");

// Finally, start the application
KafkaStreams streams = new KafkaStreams(builder, streamsConfiguration);
streams.start();