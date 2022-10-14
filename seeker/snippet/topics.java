//date: 2022-10-14T17:21:24Z
//url: https://api.github.com/gists/925e44004057d5fceeb443616652f24f
//owner: https://api.github.com/users/fspacek

 try (final var client = AdminClient.create(Map.of(CommonClientConfigs.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092"))) {
            final var groupIds = client.listConsumerGroups().all().get().
                    stream().map(ConsumerGroupListing::groupId).collect(Collectors.toList());

            //get groups
            final var groups = client.
                    describeConsumerGroups(groupIds).all().get();

            //search for topics with members
            final var topicsWithMembers = new HashSet<String>();
            for (final var groupId : groupIds) {
                final var groupDescription = groups.get(groupId);
                //find if any description is connected to the topic with topicName
                groupDescription.members().stream().
                        map(s -> s.assignment().topicPartitions()).
                        flatMap(Collection::stream).map(TopicPartition::topic).forEach(topicsWithMembers::add);
            }

            final var allTopics = client.listTopics().names().get();
            
            //remove topic with members
            allTopics.removeAll(topicsWithMembers);
            
            //print topics without any members
            allTopics.forEach(System.out::println);
        }