//date: 2022-06-01T17:01:56Z
//url: https://api.github.com/gists/284afd452fd0bc6926cc82f31f8a0eca
//owner: https://api.github.com/users/rahulrathore44

package io.testing.tables.datatables;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

import org.apache.commons.collections4.map.HashedMap;
import org.apache.poi.openxml4j.exceptions.InvalidFormatException;
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.xssf.usermodel.XSSFRow;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import io.cucumber.core.logging.Logger;
import io.cucumber.core.logging.LoggerFactory;

public class ExcelDataReader implements IDataReader {
	private final ExcelConfiguration config;
	private Logger logger = LoggerFactory.getLogger(ExcelDataReader.class);

	public ExcelDataReader(ExcelConfiguration config) {
		this.config = config;
	}
	// 1. To get the instance of work book

	private XSSFWorkbook getWorkBook() throws InvalidFormatException, IOException {
		return new XSSFWorkbook(new File(config.getFileLocation()));
	}

	// 2. Get the sheet using the work book object

	private XSSFSheet getSheet(XSSFWorkbook workBook) {
		return workBook.getSheet(config.getSheetName());
	}

	// 3. To get the header from the excel file

	private List<String> getHeaders(XSSFSheet sheet) {
		List<String> headers = new ArrayList<String>();
		XSSFRow row = sheet.getRow(0);
		row.forEach((cell) -> {
			headers.add(cell.getStringCellValue());
		});
		return Collections.unmodifiableList(headers);
	}

	/**
	 * 
	 * every row --> Map(header, column value)
	 * first_name	last_name	email	gender	city
	 * Shurlocke	Chapleo	schapleo0@usa.gov	Male	Watuweri --> Map(first_name=Shurlocke, last_name=Chapleo, email=schapleo0@usa.gov...)
	 * Hali	Allery	hallery1@blogtalkradio.com	Female	Kokemäki --> Map(first_name=Hali, last_name=Allery, email=hallery1@blogtalkradio.com...)
	 * 
	 * List<Map<String, String>> data
	 */
	@Override
	public List<Map<String, String>> getAllRows() {
		List<Map<String, String>> data = new ArrayList<Map<String, String>>();

		try (XSSFWorkbook workBook = getWorkBook()) {
			XSSFSheet sheet = getSheet(workBook);
			data = getData(sheet);
		} catch (Exception e) {
			logger.error(e, () -> {
				return String.format("Not able to read the excel %s from location %s", config.getFileName(),
						config.getFileLocation());
			});
			return Collections.emptyList();
		}
		return Collections.unmodifiableList(data);

	}

	private List<Map<String, String>> getData(XSSFSheet sheet) {
		List<Map<String, String>> data = new ArrayList<Map<String, String>>();
		List<String> headers = getHeaders(sheet);

		for (int i = 1; i <= sheet.getLastRowNum(); i++) {
			Map<String, String> rowMap = new HashedMap<String, String>();
			XSSFRow row = sheet.getRow(i);
			forEachWithCounter(row, (index, cell) -> {
				rowMap.put(headers.get(index), cell.getStringCellValue());
			});
			data.add(rowMap);
		}
		return Collections.unmodifiableList(data);
	}

	private Map<String, String> getData(XSSFSheet sheet, int rowIndex) {
		List<String> headers = getHeaders(sheet);
		Map<String, String> rowMap = new HashedMap<String, String>();

		XSSFRow row = sheet.getRow(rowIndex);
		forEachWithCounter(row, (index, cell) -> {
			rowMap.put(headers.get(index), cell.getStringCellValue());
		});

		forEachWithCounter(row, (i,j) -> {
			
		});
		
		return Collections.unmodifiableMap(rowMap);
	}

	@Override
	public Map<String, String> getASingleRow() {
		Map<String, String> data = new HashedMap<String, String>();

		try (XSSFWorkbook workBook = getWorkBook()) {
			XSSFSheet sheet = getSheet(workBook);
			data = getData(sheet, config.getIndex());
		} catch (Exception e) {
			logger.error(e, () -> {
				return String.format("Not able to read the excel %s from location %s", config.getFileName(),
						config.getFileLocation());
			});
			return Collections.emptyMap();
		}
		return Collections.unmodifiableMap(data);
	}

	private void forEachWithCounter(Iterable<Cell> source, BiConsumer<Integer, Cell> biConsumer) {
		int i = 0;
		for (Cell cell : source) {
			biConsumer.accept(i, cell);
			i++;
		}
	}

}
