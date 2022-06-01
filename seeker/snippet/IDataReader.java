//date: 2022-06-01T17:01:56Z
//url: https://api.github.com/gists/284afd452fd0bc6926cc82f31f8a0eca
//owner: https://api.github.com/users/rahulrathore44

package io.testing.tables.datatables;

import java.util.List;
import java.util.Map;

public interface IDataReader {
	
	/**
	 * To get all the rows from the excel
	 * @return
	 */
	public List<Map<String, String>> getAllRows();
	
	
	/**
	 * To get a single row from the excel
	 * @return
	 */
	public Map<String, String> getASingleRow();
}
