#date: 2022-09-16T21:13:19Z
#url: https://api.github.com/gists/23914bc95b8e376e9b24f2674f73cef9
#owner: https://api.github.com/users/jrragan

import logging
import time
from typing import Iterable, Optional, Sized, Union, List, Tuple

import xlsxwriter
from xlsxwriter.exceptions import DuplicateWorksheetName

logger = logging.getLogger(__name__)


class ExcelWriter:
    def __init__(self, filename,
                 overview_worksheet_name,
                 overview_headers=None,
                 overview_col_widths=None,
                 overview_header_row=0,
                 overview_start_row=3,
                 overview_start_column=1,
                 construct_overview=True,
                 constant_memory=False,
                 in_memory=True,
                 strings_to_numbers=False,
                 strings_to_formulas=False,
                 strings_to_urls=False,
                 use_future_functions=False,
                 max_url_length=2079,
                 nan_inf_to_errors=False,
                 ):
        """

        :param filename:
        :type filename:
        :param overview_headers:
        :type overview_headers:
        :param overview_col_widths:
        :type overview_col_widths:
        :param overview_worksheet_name:
        :type overview_worksheet_name:
        :param overview_header_row:
        :type overview_header_row:
        :param overview_start_row:
        :type overview_start_row:
        :param overview_start_column:
        :type overview_start_column:
        constant_memory: Reduces the amount of data stored in memory so that large files can be written efficiently:
            workbook = xlsxwriter.Workbook(filename, {'constant_memory': True})
            Note, in this mode a row of data is written and then discarded when a cell in a new row is added via one of the worksheet write_() methods. Therefore, once this mode is active, data should be written in sequential row order. For this reason the add_table() and merge_range() Worksheet methods don’t work in this mode.
            See Working with Memory and Performance for more details.
        tmpdir: XlsxWriter stores workbook data in temporary files prior to assembling the final XLSX file. The temporary files are created in the system’s temp directory. If the default temporary directory isn’t accessible to your application, or doesn’t contain enough space, you can specify an alternative location using the tmpdir option:
            workbook = xlsxwriter.Workbook(filename, {'tmpdir': '/home/user/tmp'})
            The temporary directory must exist and will not be created.
        in_memory: To avoid the use of temporary files in the assembly of the final XLSX file, for example on servers that don’t allow temp files, set the in_memory constructor option to True:
            workbook = xlsxwriter.Workbook(filename, {'in_memory': True})
            This option overrides the constant_memory option.
        strings_to_numbers: Enable the worksheet.write() method to convert strings to numbers, where possible, using float() in order to avoid an Excel warning about “Numbers Stored as Text”. The default is False. To enable this option use:
            workbook = xlsxwriter.Workbook(filename, {'strings_to_numbers': True})
        strings_to_formulas: Enable the worksheet.write() method to convert strings to formulas. The default is True. To disable this option use:
            workbook = xlsxwriter.Workbook(filename, {'strings_to_formulas': False})
        strings_to_urls: Enable the worksheet.write() method to convert strings to urls. The default is True. To disable this option use:
            workbook = xlsxwriter.Workbook(filename, {'strings_to_urls': False})
        use_future_functions: Enable the use of newer Excel “future” functions without having to prefix them with with _xlfn.. The default is False. To enable this option use:
            workbook = xlsxwriter.Workbook(filename, {'use_future_functions': True})
            See also Formulas added in Excel 2010 and later.
        max_url_length: Set the maximum length for hyperlinks in worksheets. The default is 2079 and the minimum is 255. Versions of Excel prior to Excel 2015 limited hyperlink links and anchor/locations to 255 characters each. Versions after that support urls up to 2079 characters. XlsxWriter versions >= 1.2.3 support the new longer limit by default. However, a lower or user defined limit can be set via the max_url_length option:
            workbook = xlsxwriter.Workbook(filename, {'max_url_length': 255})
        nan_inf_to_errors: Enable the worksheet.write() and write_number() methods to convert nan, inf and -inf to Excel errors. Excel doesn’t handle NAN/INF as numbers so as a workaround they are mapped to formulas that yield the error codes #NUM! and #DIV/0!. The default is False. To enable this option use:
            workbook = xlsxwriter.Workbook(filename, {'nan_inf_to_errors': True})

        """
        logger.info("set_excel method")
        self.indices = {overview_worksheet_name: overview_start_row}
        self.start_columns = {overview_worksheet_name: overview_start_column}
        self.header_rows = {overview_worksheet_name: overview_header_row}
        self.headers = {}
        self.column_widths = {}
        if overview_col_widths is not None:
            self.column_widths = {overview_worksheet_name: overview_col_widths[:]}
        self.workbook = xlsxwriter.Workbook(filename, {'strings_to_numbers': strings_to_numbers,
                                                       'strings_to_formulas': strings_to_formulas,
                                                       'strings_to_urls': strings_to_urls,
                                                       'constant_memory': constant_memory,
                                                       'in_memory': in_memory,
                                                       'use_future_functions': use_future_functions,
                                                       'max_url_length': max_url_length,
                                                       'nan_inf_to_errors': nan_inf_to_errors})
        self.header_format = self.workbook.add_format({'bold': True, 'align': 'center', 'valign': 'center'})
        self.link_format = self.workbook.add_format({'color': 'blue', 'underline': 1})
        self.link_format1 = self.workbook.add_format({'color': 'blue', 'underline': 1, 'bg_color': 'gray'})
        self.overview = self.workbook.add_worksheet(overview_worksheet_name)
        self.worksheets = {overview_worksheet_name: self.overview}
        self.worksheet = self.overview
        self.current_worksheet_name = overview_worksheet_name
        self.overview_worksheet_name = overview_worksheet_name
        if construct_overview:
            self.add_headers_to_worksheet(overview_worksheet_name, overview_headers, overview_header_row,
                                          overview_start_column)

    @staticmethod
    def _check_for_none(device_param):
        if device_param is None:
            logger.debug("check for none: {}".format(device_param))
            return " "
        return device_param

    def add_excel_worksheet(self, worksheet_name: str, worksheet_start_row: int = 0, worksheet_start_col: int = 0,
                            header_row: Optional[int] = 0, column_widths: Optional[Iterable] = None):
        if column_widths is not None:
            self.column_widths[worksheet_name] = column_widths
        self.header_rows[worksheet_name] = header_row
        try:
            self.worksheets[worksheet_name] = self.workbook.add_worksheet(worksheet_name)
            self.indices[worksheet_name] = worksheet_start_row
            self.start_columns[worksheet_name] = worksheet_start_col
        except DuplicateWorksheetName as e:
            logger.error(f"{worksheet_name} Already Exists")
            logger.debug(e)
            raise

    def add_headers_to_worksheet(self, worksheet_name: str, headers: Iterable, header_row: int, start_col: int,
                                 header_format: Optional[dict] = None):
        self.header_rows[worksheet_name] = header_row
        self.headers[worksheet_name] = headers

        if header_format is None:
            header_format = self.header_format
        for c, header in enumerate(headers, start=start_col):
            self.worksheets[worksheet_name].write_string(header_row, c, header, header_format)

    def set_excel_worksheet(self, worksheet_name: str):
        try:
            self.worksheet = self.worksheets[worksheet_name]
            self.current_worksheet_name = worksheet_name
        except KeyError as e:
            logger.error(f"{worksheet_name} Does Not Exist")
            logger.debug(e)
            raise

    def write_excel_row(self, row: Iterable, row_index=None, front_page_only=True):
        # logger.info("write_excel_row method")
        if not front_page_only:
            pass
        if not front_page_only:
            pass
        else:
            if row_index is None:
                self.worksheet.write_row(self.indices[self.current_worksheet_name],
                                         self.start_columns[self.current_worksheet_name],
                                         row)
                self.indices[self.current_worksheet_name] += 1
            else:
                self.worksheet.write_row(row_index, self.start_columns[self.current_worksheet_name], row)

        if not front_page_only:
            pass

    def add_table_to_worksheet(self, worksheet_name: str, table_style: str = "Table Style Medium 15",
                               header_format: Optional[dict] = None):
        if header_format is None:
            header_format = self.header_format
        self.worksheets[worksheet_name].add_table(self.header_rows[worksheet_name],
                                                  self.start_columns[worksheet_name],
                                                  self.indices[worksheet_name] - 1,
                                                  self.start_columns[worksheet_name] + len(
                                                      self.headers[worksheet_name]) - 1,
                                                  {"style": table_style,
                                                   "columns": [{'header': x, 'header_format': header_format} for x in
                                                               self.headers[worksheet_name]]})
        for col, width in enumerate(self.column_widths.get(worksheet_name,
                                                           [len(w) for w in self.headers[worksheet_name]]),
                                    start=self.start_columns[worksheet_name]):
            self.worksheets[worksheet_name].set_column(col, col, width)

    def add_table_with_data_to_worksheet(self, worksheet_name: str,
                                         data: Iterable[Iterable],
                                         headers: Union[List, Tuple],
                                         column_widths: Optional[Iterable] = None,
                                         header_row: bool = True,
                                         first_column: bool = True,
                                         last_column: bool = False,
                                         total_row: bool = False,
                                         table_style: str = "Table Style Medium 15",
                                         header_format: Optional[dict] = None):
        if header_format is None:
            header_format = self.header_format
        if column_widths is not None:
            self.column_widths[worksheet_name] = column_widths
        self.worksheets[worksheet_name].add_table(self.header_rows[worksheet_name],
                                                  self.start_columns[worksheet_name],
                                                  len(data),
                                                  self.start_columns[worksheet_name] + len(headers) - 1,
                                                  {"data": data, "header_row": header_row,
                                                   "first_column": first_column,
                                                   "last_column": last_column,
                                                   "total_row": total_row,
                                                   "style": table_style,
                                                   "columns": [{'header': x, 'header_format': header_format} for x in
                                                               headers]})
        for col, width in enumerate(self.column_widths.get(worksheet_name,
                                                           [len(w) + 4 for w in headers]),
                                    start=self.start_columns[worksheet_name]):
            self.worksheets[worksheet_name].set_column(col, col, width)

    def finish(self, skip_table=False):
        if not skip_table:
            print("adding table to worksheet at finish")
            self.add_table_to_worksheet(self.overview_worksheet_name)
        print("closing workbook")
        start = time.time()
        self.workbook.close()
        print(f"Time to close workbook is {time.time() - start} seconds")


if __name__ == "__main__":
    headers = ['Sauropodomorpha', 'Theropoda', 'Thyreophora', 'Marginocephalia', 'Ornithopoda']
    row1 = ['Plateosaurus', 'Tyrannosaurus', 'Stegosarus', 'Triceratops', 'Heterodontosaurus']
    row2 = ['Diplodicus', 'Carnotaurus', 'Ankylosaurus', 'Torosaurus', 'Edmontosaurus']
    writer = ExcelWriter('prehistoric_table.xlsx', 'dinosaurs', overview_start_row=1, construct_overview=False)
    # writer.write_excel_row(row1)
    # writer.write_excel_row(row2)
    writer.add_table_with_data_to_worksheet('dinosaurs', [row1, row2], headers, first_column=False)
    headers = ['Pelycosaurs', 'Dinocephalia', 'Dicynodontia', 'basal Cynodonts', 'Mammals', 'Gorgonopsia']
    row1 = ['Edaphosaurus', 'Moschops', 'Diictodon', 'Thrinaxodon', 'Andrewsarchus', 'Inostrancevia']
    row2 = ['Dimetrodon', 'Anteosaurus', 'Placerias', 'Procynosuchus', 'Brontotherium', 'Gorgonops']
    writer.add_excel_worksheet('Synapsids', 1, 1)
    # writer.add_headers_to_worksheet('Synapsids', headers, 0, 1)
    # writer.set_excel_worksheet('Synapsids')
    # writer.write_excel_row(row1)
    # writer.write_excel_row(row2)
    writer.add_table_with_data_to_worksheet('Synapsids', [row1, row2], headers, table_style='Table Style Light 11',
                                            first_column=False)
    writer.finish(skip_table=True)
    print("Finished!")
