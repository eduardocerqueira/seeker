//date: 2021-11-24T16:55:38Z
//url: https://api.github.com/gists/5b8fc71db0d6ad9fa2f46c6212cb8dad
//owner: https://api.github.com/users/AlexMakesSoftware

    public HashMap<EdgeAreaKey,Boolean> load(File f) {
        this.lookup = new HashMap<>();
        CSVFormat format = CSVFormat.Builder.create(CSVFormat.RFC4180)
            .setHeader(Fields.class)
            .setSkipHeaderRecord(true)
            .setAllowMissingColumnNames(false)
            .setIgnoreEmptyLines(true)
            .setIgnoreSurroundingSpaces(true)
            .setQuoteMode(QuoteMode.NON_NUMERIC)
            .build();        
        
		try (CSVParser parser = new CSVParser(new FileReader(f), format)) {
            List<CSVRecord> csvRecords = parser.getRecords();

            if(csvRecords.size()==0) throw new RuntimeException("no data in file "+f);
			
            List<String> headers = parser.getHeaderNames();
            if(!headers.toString().equals(EXPECTED_HEADER_EVALS_TO)) {
                throw new RuntimeException("bad header in edge file:"+f); }
            
            for (CSVRecord r : csvRecords) {
                int county = Integer.parseInt(r.get(Fields.COUNTY_ID));
                int year = Integer.parseInt(r.get(Fields.YEAR));
                
                if(year>lastYear) lastYear = year;
                if(year<firstYear) firstYear = year;

                Boolean isEdge = Boolean.valueOf(r.get(Fields.IS_EDGE));
                lookup.put(new EdgeAreaKey(year, county), isEdge);
            }
		} catch (IOException e) {
			throw new RuntimeException("Error loading edge file:"+f);
		} 

        validationChecks();

        //TODO: later on, when we upgrade Java, make this an immutable collection.
        return lookup;
    }