//date: 2024-12-17T17:04:03Z
//url: https://api.github.com/gists/c29939887bbe34aed0e503bc2e7dfaf6
//owner: https://api.github.com/users/cheparsky

public class ExcelUtility {
    
    private final String filePath;
    private final Workbook workbook;
    private final Sheet sheet;

    public ExcelUtility(String filePath, String sheetName) throws IOException {
        try (FileInputStream fis = new FileInputStream(new File(filePath))) {
            this.filePath = filePath;
            this.workbook = new XSSFWorkbook(fis);
            this.sheet = workbook.getSheet(sheetName);
            if (this.sheet == null) {
                throw new IllegalArgumentException("Sheet '" + sheetName + "' not found!");
            }
        }
    }
  
    public String getCellData(int rowNumber, int columnNumber) {
        Row row = sheet.getRow(rowNumber);
        if (row == null) return null;
        Cell cell = row.getCell(columnNumber);
        if (cell == null) return null;

        // Return value as it is displayed in Excel file
        DataFormatter formatter = new DataFormatter();
        return formatter.formatCellValue(cell);
    }
    
    public ExcelUtility updateCell(int rowNumber, int columnNumber, String value) {
        Row row = sheet.getRow(rowNumber);
        if (row == null) {
            row = sheet.createRow(rowNumber);
        }
        Cell cell = row.getCell(columnNumber, Row.MissingCellPolicy.CREATE_NULL_AS_BLANK);
        cell.setCellValue(value);
        return this;
    }
    
    public void save() throws IOException {
        try (FileOutputStream fos = new FileOutputStream(new File(filePath))) {
            workbook.write(fos);
        }
    }
        
    public void close() throws IOException {
        if (workbook != null) {
            workbook.close();
        }
    }
    
}