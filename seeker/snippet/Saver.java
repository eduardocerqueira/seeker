//date: 2021-11-24T16:54:16Z
//url: https://api.github.com/gists/f942485e8d072dc4c2e2f687ff604385
//owner: https://api.github.com/users/AlexMakesSoftware


    public void save(File f) {
		CSVFormat format = CSVFormat.Builder.create(CSVFormat.RFC4180)
            .setHeader(Fields.class)            
            .setRecordSeparator(System.lineSeparator())
            .setTrailingDelimiter(false)
            .setAutoFlush(true)
            .setQuoteMode(QuoteMode.NON_NUMERIC)
            .build();        
		CSVPrinter csvPrinter = null;
		try {
			csvPrinter = new CSVPrinter(new FileWriter(f), format);
            for(int area = 1;area<=MAX_AREA_ID;area++) {
                for(int year = START_YEAR;year<=END_YEAR;year++) {                    
                    csvPrinter.print(area);
                    csvPrinter.print(year);
                    Boolean isEdge = lookup.get(new EdgeAreaKey(year, area));                    
                    if(isEdge==null) isEdge = Boolean.FALSE;
                    csvPrinter.print(isEdge.toString());
                    csvPrinter.println();
                }                
            }
			csvPrinter.flush();			
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (csvPrinter != null) {
				try { csvPrinter.close();
				} catch (IOException e) { //na
                }
			}
		}
    }