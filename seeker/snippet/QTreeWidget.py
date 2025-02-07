#date: 2025-02-07T16:45:58Z
#url: https://api.github.com/gists/95c0615a6607af25f6810b12e088a709
#owner: https://api.github.com/users/gregoiredehame

from .. import qt

class Node():
    def __init__(self, name=""):
        self.name = name
        self.length = len(name)


class TreeNode(qt.QTreeWidgetItem):
    def __init__(self, parent:qt.QTreeWidgetItem=None, data:Node=None):
        """Custom QTreeWidgetItem representing a tree node.

        Args:
            parent (qt.QTreeWidgetItem): - The parent item of this tree node.
            data                 (Node): - The data object associated with this node.
        """
        super(TreeNode, self).__init__(parent)
        self.data = data


    @property
    def data(self) -> Node:
        """Property to get the data associated with this TreeNode.

        Returns:
            Node: - The data object stored in this node.
        """
        return self._data


    @data.setter
    def data(self, value:Node=None) -> None:
        """Property setter to update the node's data.

        Args:
            value (Node): - The new data to assign to the TreeNode.
        """
        self._data = value
        for i in range(int(self.data.length)):
            self.setText(i, self.data.name[i])


    def update(self) -> None:
        """Function that will update the displayed text of the TreeNode."""
        if self.data:
            self.setText(0, self.data.name)
            
            
    def childrens(self, top_level_only:bool=False, column:int=None) -> list:
        """Returns a list of child QTreeWidgetItems.

        Args:
            top_level_only:  (bool): - if True, only return direct children, otherwise return all nested children.
            column: (int, optional): - if set, returns only text from the given column. Defaults to None.
        """
        if top_level_only:
            items = [self.child(i) for i in range(self.childCount())]
        else:
            def get_subtree_nodes(item):
                return [item] + [node for i in range(item.childCount()) for node in get_subtree_nodes(item.child(i))]

            items = [node for i in range(self.childCount()) for node in get_subtree_nodes(self.child(i))]

        return [item.text(column) for item in items] if isinstance(column, int) else items  


class TreeWidget(qt.QTreeWidget):
    widgetIsResizing = qt.signal(object)
    itemOrderChanged = qt.signal(object)
    itemHeaderClicked = qt.signal(object)
    itemRenamed = qt.signal(tuple)
    itemRemoved = qt.signal(object)
    def __init__(self, parent:qt.QWidget=None, headers:tuple=None, drag_and_drop:bool=True, drag_in:bool=True, header_clickable:bool=True, header_movable:bool=False, height:int=5, height_resizable:bool=False, columns_renamable:list=None, removable:bool=False):        
        """Advanced Custom class for pyside QTreeWidget in order to allow custom functions here.
        
        Args:
            parent:      (qt.QWidget): - parent QWidget.
            headers:     (tuple/list): - headers names ex: ("Name", "Shapes", "Tag")
            drag_and_drop:     (bool): - true, will allow the use to drag and drop ( change QTreeWidgetItem order ) will emit itemHeaderClicked
            drag_in:           (bool): - true, will allow user to drag QTreeWidgetItem inside each other. will emit itemHeaderClicked
            header_clickable:  (bool): - true, will allow user to change order of QTreeWidgetItems per column names
            header_movable:    (bool): - true, will allow user to change the order of the colmuns
            height:             (int): - integer value for QTreeWidgetItem(s) height
            height_resizable:  (bool): - true, will allow user to scroll with middle mouse button for resizing QTreeWidgetItem(s) height
            columns_renamable: (list): - list of integer that will allowed the column to be renamable this will emit itemRenamed
            removable:         (bool): - true, will allowed user to remove items.
        """
        qt.QTreeWidget.__init__(self, parent)
        self.drag_and_drop = drag_and_drop
        self.drag_in = drag_in
        self.header_clicked = header_clickable
        self.header_movable = header_movable
        self.height_resizable = height_resizable
        self.row_height = height
        self.columns_renamable = columns_renamable
        self.removable = removable

        self.setSelectionMode(qt.QAbstractItemView.ExtendedSelection)
        self.setItemsExpandable(True)
        self.setExpandsOnDoubleClick(False)
        self.setAnimated(True)
        self.setDragEnabled(True) if self.drag_and_drop else self.setDragEnabled(False)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(qt.QAbstractItemView.InternalMove)
        self.setAlternatingRowColors(True)
        self.setHeaderHidden(True)
        
        if headers and isinstance(headers, (list, tuple)):
            self.setHeaders(headers)

        self.itemExpanded.connect(self.onItemExpanded)
        self.itemCollapsed.connect(self.onItemCollapsed)

        self.setHeight(self.row_height)         
                 
                 
    def setHeight(self, height:int=None) -> None:
        """Function that will set the QTreeWidgetItem(s) height size
        Args:
            height: (int): - integer pixel value to set QTreeWidgetItem(s) height
        """
        if isinstance(height, int):
            self.setStyleSheet(f"QTreeView::item {{ padding: {height}px; }}")
  
  
    def setHeaders(self, headers:list=None) -> None:
        """function that will set a list of string as QTreeWidget's headers
        
        Args:
            headers: (list): - list of string to use as titles inside header
        """
        if headers and isinstance(headers, (list, tuple)):
            if len(headers) > 0:
                
                self.setHeaderHidden(False)
                self.setColumnCount(len(headers))
                self.setHeaderLabels(headers)
                if len(headers) <= 1:
                    self.header().setStretchLastSection(True)
                else:
                    self.header().setSectionsMovable(self.header_movable)
                    total_width = self.width()
                    for i in range(len(headers)):
                        self.setColumnWidth(i, total_width // len(headers))
                        self.header().resizeSection(i, total_width // len(headers))  
                self.header().setSectionsClickable(self.header_clicked)
                self.setSortingEnabled(self.header_clicked)  
                
                
    def onItemExpanded(self, item:qt.QTreeWidgetItem=None) -> None:
        """function that will override onItemExpanded.
        we want to make sure we also expand all childrens while shift is pressed.
        
        Args:
            item: (qt.QTreeWidgetItem): - QTreeWidgetItem expanded
        """
        if qt.QApplication.keyboardModifiers() == qt.Qt.ShiftModifier:
            self.expandItem(item)
            [self.onItemExpanded(item.child(i)) for i in range(item.childCount())]
        
        
    def onItemCollapsed(self, item:qt.QTreeWidgetItem=None) -> None:
        """function that will override onItemCollapsed.
        we want to make sure we also expand all childrens while shift is pressed.
        
        Args:
            item: (qt.QTreeWidgetItem): - QTreeWidgetItem collapsed
        """
        if qt.QApplication.keyboardModifiers() == qt.Qt.ShiftModifier:
            self.collapseItem(item)
            [self.onItemCollapsed(item.child(i)) for i in range(item.childCount())] 
            
            
    def headerClicked(self, index:int=None) -> None:
        """function that will override headerCliked command, and will make sure we emit
        
        Args:
            index: (int): - index for column header
        """
        if self.header_clicked:
            column_name = self.headerItem().text(index)
            self.itemHeaderClicked.emit(index)
            
        
    def childrens(self, top_level_only:bool=False, column:int=None) -> list:
        """function that will list the QTreeWidget childrens 
        
        Args:
            top_level_only: (bool): - True, will only return the top level items, otherwhise returns all
            column:          (int): - integer to return only text from a specific column
        """
        if top_level_only:
            items = [self.topLevelItem(i) for i in range(self.topLevelItemCount())]
            
        else:
            def get_subtree_nodes(item:qt.QWidgetItem=None) -> list:
                return [item] + [node for i in range(item.childCount()) for node in get_subtree_nodes(item.child(i))]
            items = [node for i in range(self.topLevelItemCount()) for node in get_subtree_nodes(self.topLevelItem(i))]
            
        return [item.text(column) for item in items] if isinstance(column, int) else items
        
        
    def removeItems(self, items:list=None) -> None:
        """function that will remove the given items, and will go through all given items and will
        also remove the given QTreeWidgetItems childrens
        
        Args:
            items: (list): - list of qt.QTreeWidgetItems to remove
        """
        if isinstance(items, (tuple, list)):
            for item in items:
                if not item or not isinstance(item, qt.QTreeWidgetItem):
                    continue
                    
                while item.childCount() > 0:
                    self.removeItems([item.child(0)])
                    
                parent = item.parent()
                if parent:
                    parent.takeChild(parent.indexOfChild(item))
                else:
                    self.takeTopLevelItem(self.indexOfTopLevelItem(item))
                
                self.itemRemoved.emit(item)
                    
                    
    def closeEditor(self, editor:qt.QTextEdit=None, item:qt.QTreeWidgetItem=None, column:int=None, update:bool=None) -> None:
        """function that will close the QTextEdit and will emit a proper tuple as itemRenamed
        
        emit:
            (item, <qt.QTreeWidgetItem>, column <int>, new_name <str>, previous_name <str> )
        
        Args:
            editor:     (qt.QTextEdit): - QTextEdit where to query current new name
            item: (qt.QTreeWidgetItem): - QTreeWidgetItem to emit
            column:              (int): - integer value of the column
            update:             (bool): - boolean, true will update
        """
        if update and column in self.columns_renamable:
            self.itemRenamed.emit((item, column, editor.toPlainText(), item.text(column)))
        editor.deleteLater()
                
                
    # QtCore.QEvent               
    def resizeEvent(self, event:qt.QtCore.QEvent=None) -> None:
        """function that will override resize QEvent in order to emit a widgetIsResizing
        
        Args:
            event: (qt.QtCore.QEvent): - resize QEvent
        """
        self.widgetIsResizing.emit(event)
        super().resizeEvent(event)
                
                   
    def wheelEvent(self, event:qt.QtCore.QEvent=None) -> None:
        """function that will override wheel QEvent in order to resize dynamically the QTreeWidgetItem(s) height
        this will work only if QTreeWidget's has been initialize with height_resizable on 
        
        Args:
            event: (qt.QtCore.QEvent): - wheel QEvent
        """
        if self.height_resizable and qt.QApplication.keyboardModifiers() == qt.Qt.ControlModifier:
            self.row_height = max(1, self.row_height + (1 if event.angleDelta().y() > 0 else -1))
            self.setHeight(self.row_height)
            event.accept()
        super(TreeWidget, self).wheelEvent(event)
            

    def keyPressEvent(self, event:qt.QtCore.QEvent=None) -> None:
        """function that will override keyPressEvent
        the idea is to allowed user to remove an item when delete is pressed.
        
        Args:
            event: (qt.QtCore.QEvent): - keyPressEvent
        """
        if event.key() in (qt.Qt.Key_Delete, qt.Qt.Key_Backspace) and self.removable:
            selected_items = self.selectedItems()
            if selected_items:
                message = qt.QMessageBox(self)
                message.setIcon(qt.QMessageBox.Warning)
                message.setWindowTitle("Remove Items")
                message.setText("Do you really want to remove selected Item(s)?             ")
                message.setInformativeText("This action is undoable.")
                message.setStandardButtons(qt.QMessageBox.Yes | qt.QMessageBox.No | qt.QMessageBox.Cancel)
                if qt.QMessageBox.critical(self, "Remove Items", "Do you really want to remove selected Item(s)?             ", qt.QMessageBox.Yes | qt.QMessageBox.No | qt.QMessageBox.Cancel, qt.QMessageBox.No) == qt.QMessageBox.Yes:
                    self.removeItems(selected_items)
  
        elif (event.key() == qt.Qt.Key_Escape and event.modifiers() == qt.Qt.NoModifier):
            self.clearSelection()
            
        super(TreeWidget, self).keyPressEvent(event)


    def mousePressEvent(self, event:qt.QtCore.QEvent=None) -> None:
        """function that will override mousePressEvent in order to clear selection if nothing is selected.
        
        Args:
            event: (qt.QtCore.QEvent): - mousePressEvent
        """
        if self.itemAt(event.pos()) is None:
            self.clearSelection()
        super(TreeWidget, self).mousePressEvent(event)
        
        
    def mouseDoubleClickEvent(self, event:qt.QtCore.QEvent=None) -> None:
        """function that will override mouseDoubleClickEvent.
        this will be a bit more advanced, since it will allowed the user to pop a textField on the clicked item's column
        this will not rename by default, but will emit itemRenamed in case we just want to copy and paste
        
        Args:
            event: (qt.QtCore.QEvent): - mouseDoubleClickEvent
        """
        if self.columns_renamable and isinstance(self.columns_renamable, (list, tuple)):
            item = self.itemAt(event.pos())
            if item:
                column = self.header().logicalIndexAt(event.pos().x())
                if column in self.columns_renamable:
                    rect = self.visualItemRect(item)
                    editor = TreeTextEdit(self.viewport(), self, item, column)
                    editor.setGeometry(sum(self.columnWidth(i) for i in range(column)) - self.horizontalScrollBar().value(), rect.y(), self.columnWidth(column), rect.height())
                    editor.setText(item.text(column).lstrip('.'))
                    editor.setFocus()
                    editor.show()
                    editor.selectAll()
                    editor.focusOutEvent = lambda event: self.closeEditor(editor, item, column, False)
        super(TreeWidget, self).mouseDoubleClickEvent(event)


    def dragEnterEvent(self, event:qt.QtCore.QEvent=None) -> None:
        """function that will override dragEnterEvent QEvent, this idea here is to lock dragging a
        QTreeWidgetItem inside another one if QTreeWidget's drag_in is False
        
        Args:
            event: (qt.QtCore.QEvent): - dragEnterEvent
        """
        if self.drag_and_drop:
            item = self.itemAt(event.pos())   
            if item is not None and (isinstance(item.data, Node)):
                event.acceptProposedAction()
            else:
                super(TreeWidget, self).dragEnterEvent(event)


    def dragMoveEvent(self, event:qt.QtCore.QEvent=None) -> None:
        """function that will override dragMoveEvent QEvent, this idea here is to lock dragging a
        QTreeWidgetItem inside another one if QTreeWidget's drag_and_drop is False
        
        Args:
            event: (qt.QtCore.QEvent): - dragMoveEvent
        """
        if self.drag_and_drop:
            item = self.itemAt(event.pos())
            super(TreeWidget, self).dragMoveEvent(event)


    def dropEvent(self, event:qt.QtCore.QEvent=None) -> None:
        """function that will override dropEvent QEvent, here, we will just emit itemOrderChanged if
        the order changed on QTreeWidget
        
        Args:
            event: (qt.QtCore.QEvent): - dropEvent
        """
        if self.drag_and_drop:
            item = self.itemAt(event.pos())
            if item is not None and (isinstance(item.data, Node)):
                if not self.drag_in:
                    if self.dropIndicatorPosition() == self.OnItem or self.dropIndicatorPosition() == self.OnViewport:
                        pass
                        
                    else:
                        super(TreeWidget, self).dropEvent(event)
                        self.itemOrderChanged.emit(True)
                else:
                    super(TreeWidget, self).dropEvent(event)
                    self.itemOrderChanged.emit(True)
            else:
                event.setDropAction(qt.Qt.IgnoreAction)
            
          
          
class TreeTextEdit(qt.QTextEdit):
    """Custom QTextEdit class in order to rename double clicked items.
    by default QTreeWidgetItems are not renamable, and that's an option we wanna have.
    this will emit "itemRenamed" when user hit "enter", which is not possible by default using the focusOutEvent.
    """
    def __init__(self, parent=None, widget=None, item=None, column=None):
        super().__init__(parent)
        self.widget = widget
        self.item = item
        self.column = column


    def keyPressEvent(self, event:qt.QtCore.QEvent=None) -> None:
        """function that will override keyPressEvent QEvent, here, and will allowed emitting 
        if enter is pressed.
        
        Args:
            event: (qt.QtCore.QEvent): - keyPressEvent
        """
        if event.key() in (qt.Qt.Key_Return, qt.Qt.Key_Enter):
            self.widget.closeEditor(self, self.item, self.column, True)
        else:
            super().keyPressEvent(event)