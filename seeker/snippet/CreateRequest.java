//date: 2022-11-14T17:08:05Z
//url: https://api.github.com/gists/19b51ad368106dffcffafc4afa058929
//owner: https://api.github.com/users/aleksandr-dudko

// create the Record object:
Record record = new Record();
// add the test data.
record.addColumn("sepal length (cm)", 1.2);
record.addColumn("sepal width (cm)", 2.4);
record.addColumn("petal length (cm)", 3.3);
record.addColumn("petal width (cm)", 4.1);   
// add it to RecordSet.
RecordSet recordSet = new RecordSet();
recordSet.addRecord(record);

// create the RequestBody object and add the recordSet object with property "data".
RequestBody requestBody = new RequestBody();
requestBody.addParameter("data", recordSet);