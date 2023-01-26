//date: 2023-01-26T16:52:13Z
//url: https://api.github.com/gists/05d3ae2142185de86bfac1ed84b03628
//owner: https://api.github.com/users/mstahv

        ArrayList<String> values = new ArrayList<>();
        values.add("foo");
        values.add("bar");
        ComboBox<String> stringComboBox = new ComboBox<>();

        stringComboBox.setItemsWithFilterConverter((CallbackDataProvider.FetchCallback<String, String>) query -> {
            if(query.getFilter().isPresent()) {
                String filter = query.getFilter().get();
                System.out.println("Do something else with filter: " + filter);
                return values.stream()
                        .filter(s -> s.startsWith(filter.toString()))
                        .skip(query.getOffset()).limit(query.getPageSize());
            }
            return values.stream().skip(query.getOffset()).limit(query.getPageSize());
        }, s -> s);

        stringComboBox.addValueChangeListener(v -> {
            // Selecting existing values
        });
        stringComboBox.addCustomValueSetListener(v -> {
            // adding new values (probably all intercepted already in lazy data binding)
        });
