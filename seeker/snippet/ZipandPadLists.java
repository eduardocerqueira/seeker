//date: 2025-02-20T17:01:14Z
//url: https://api.github.com/gists/27b0f013a5e8a7b707fd72bcadb68ed4
//owner: https://api.github.com/users/imsurajmishra

 public <T,U> List<List<Object>> zipAndPad(List<T> customerIds, List<U> customerNames){
      int size = Math.min(customerNames.size(), customerIds.size());
      List<List<Object>> zippedList = new ArrayList<>();
      
      // add all the elements with same size.
      for(int i=0; i<size; i++){
          zippedList.add(List.of(customerIds.get(i), customerNames.get(i)));
      }

      // if remaining elements in either list
      if(Math.max(customerNames.size(), customerIds.size()) != size) {
          remainingElements(customerIds, customerNames, size, zippedList);
      }

      return zippedList;
 }

 private <T,U> void remainingElements(List<T> customerIds, List<U> customerNames, int size, List<List<Object>> zippedList) {
     for (int j = size; j < Math.max(customerIds.size(), customerNames.size()); j++) {
          Object id = j < customerIds.size() ? customerIds.get(j) : "";
          Object name = j < customerNames.size() ? customerNames.get(j) : "";
          zippedList.add(List.of(id, name));
     }
  }

@Test
 public void givenTwoLists_WhenPerformedZipped_ThenItZipsThem(){
      ZipImplementation zipImplementation = new ZipImplementation();
      List<Integer> customerIds = List.of(1, 2);
      List<String> customerNames = List.of("sam", "den", "lenny");

      List<List<Object>> zip = zipImplementation.zip(customerIds, customerNames);
      assertEquals(List.of(List.of(1, "sam"), List.of(2, "den"), List.of("", "lenny")), zip);

      List<Integer> customerIds1 = List.of(1, 2, 3);
      List<String> customerNames1 = List.of("sam", "den", "lenny");
      List<List<Object>> zip1 = zipImplementation.zip(customerIds1, customerNames1);
      assertEquals(List.of(List.of(1, "sam"), List.of(2, "den"), List.of(3, "lenny")), zip1);
 }