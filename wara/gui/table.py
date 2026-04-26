from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt


class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data
        self.highlighted_rows = []

    def data(self, index, role):
        row = index.row()
        column = index.column()
        if role == Qt.DisplayRole:
            value = self._data.iloc[row, column]
            return str(value)
        elif role == Qt.BackgroundRole and row in self.highlighted_rows:
            return QtGui.QColor(Qt.yellow)

    def set_highlighted_rows(self, rows):
        self.highlighted_rows = rows
        self.dataChanged.emit(
            self.index(0, 0), self.index(self.rowCount(0), self.columnCount(0))
        )

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])

            if orientation == Qt.Vertical:
                return str(self._data.index[section])

    def sort(self, Ncol, order):
        """Sort table by given column number."""
        try:
            self.layoutAboutToBeChanged.emit()
            self._data = self._data.sort_values(
                self._data.columns[Ncol], ascending=not order
            )
            self.layoutChanged.emit()
        except Exception as e:
            print(e)
