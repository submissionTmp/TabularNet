# -*- coding: utf-8 -*-


class TextTable(object):
    def __init__(self):
        self.num_row = 0
        self.num_col = 0
        self.cell_matrix = None

    def __init__(self, json_table):
        self.num_row = json_table["RowCount"]
        self.num_col = json_table["ColumnCount"]
        self.cell_matrix = [[c["Text"] for c in row]
                            for row in json_table["CellAttributes"]]
