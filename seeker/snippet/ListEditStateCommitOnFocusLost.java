//date: 2022-02-10T16:56:02Z
//url: https://api.github.com/gists/447344183e017537c21f7905a062396d
//owner: https://api.github.com/users/kleopatra

/*
 * Created 02.05.2021
 */

package control.cell.edit;


import javafx.application.Application;
import javafx.beans.Observable;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.EditState;
import javafx.scene.control.EditState.EditAction;
import javafx.scene.control.Label;
import javafx.scene.control.ListView;
import javafx.scene.control.Location.IndexedLocation;
import javafx.scene.control.MenuItem;
import javafx.scene.control.Tab;
import javafx.scene.control.TabPane;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.TextField;
import javafx.scene.control.cell.TextFieldListCell;
import javafx.scene.control.cell.TextFieldTableCell;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyCodeCombination;
import javafx.scene.input.KeyCombination;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.FlowPane;
import javafx.scene.layout.HBox;
import javafx.stage.Stage;

/**
 * Visual test for testing commitOnFocusLost (ListView only) against draft
 * with EditState.
 *
 *
 * It has
 *
 * - enriched edit location property: EditState has both the location and the
 *   type of the state change
 * - ListView: having the EditState property (and keeping it in sync with old editingIndex)
 * - ListView: complete edit state change api (start, commit, cancel, stop)
 * - Cell: extracted handler on focusLost for override in subs, api to access the editedValue
 * - ListCell: using new ListView api - listening to editState, use edit api, overridden focusedLost
 *   to delegate to ListView.stopEdit
 * - ListCellBehavior: use start/stopEdit of listView
 * - TextFieldTableCell: implement editedValue
 *
 *
 * branch: https://github.com/kleopatra/jfx/tree/dokeep-edit-api-editstate
 *
 * -------------
 *
 *
 * scenarios (default): note that the fix requires the default action in ListView
 *    to be set to commit (otherwise it behaves the same as previously)
 *
 * Note: "with fix" currently means all of the above (ListView only)
 *
 * A: edit selected item -> start edit next/prev with external key (F1)
 * expected: edit committed, next/prev editing
 *    with fix: edit committed
 *    without fix: edit cancelled
 *
 * A1: not relevant ? same as A, but now code moving selection before editing
 *   with fix: edit cancelled (due to cancel from focus listener in cell?)
 *   without: edit cancelled
 *
 * A2: not relevant ? same as A, but now code moving selection after editing
 *   with fix: working as expected
 *   without: edit cancelled
 *
 * B: edit selected item, click into other not-empty cell
 * expected: edit committed, (cell selected/focused?)
 *   with fix: edit committed (selection on clicked/focus unchanged)
 *   without: edit cancelled (selection/focus on that cell?)
 *
 * B1: edit selected item, click into other empty cell
 * expected: edit committed
 *   with fix: text field looses focus with content unchanged, cell appears "selected" (blue background)
 *   without: same as with
 *
 * C: edit focused cell (no selection), move focus
 * expected: edit committed
 *    with fix: edit committed, old index selected/list (or tab) focused?
 *    without: edit cancelled, old index selected
 *
 * C1: edit focused cell (no selection), move selection
 * expected: edit committed
 *    with fix: edit committed, new cell focused and selected
 *    without: edit cancelled, new index selected
 *
 * D: edit selected cell, move focus
 * expected: edit committed
 *    with fix: edit committed, old index selected/list (or tab) focused?
 *    without: edit cancelled, old index selected
 *
 * D1: edit selected cell, move selection
 * expected: edit committed
 *    with fix: edit committed, new cell selected /list (or tab) focused?
 *    without: edit cancelled
 *
 * D2 Note: - spurious problem with selectNext? focus not moved as well?
 * if so:
 *    with fix: ??
 *    without: cell still editing
 *
 * E: override commit to edit next if possible
 *    with fix: edit committed, next not started (but: thought I have seen it work ...)
 *       it's cancelled from cell resetting cell index in commit
 *    reason for the not working: skin cancels the edit on receiving a change from the items
 *
 * ---- ListView
 *
 * scenario: move focus to component outside of control
 * - expected: edit stopped and committed
 * - actual: control still editing
 * might be incorrect expectation? yes, probably - but can be implemented by a listener
 * to scene focusOwner and manually stop if new owner is not child of the control
 *
 * To reproduce
 * - edit cell, type something
 * - focus one of the dummies below
 * - expected: editing stopped (at least) and new value committed (bug)
 * - actual: cell still editing
 *
 */
public class ListEditStateCommitOnFocusLost extends Application {

    private TabPane tabPane;
    private Scene scene;

    // special casing listView
    ListView<String> listView;

    // flags to control edits
    // overridden commit, (selects next and?) starts editing on next
    // set to true for scenario E
    // note: setting unsuitable for editNext/prev manual methods and focus/selectNext
    private boolean editNextOnCommit = false;
    // overridden commit, append new item and start editing
    private boolean appendItemAndEditOnCommit = false;
    // unused?
    // move selection in editNext/previous (before starting edit)
    private boolean moveSelectionBefore = false;
    private boolean moveSelectionAfter = false;

    @Override
    public void start(Stage primaryStage) {
        // tabPane for all
        tabPane = new TabPane(
                new Tab("List", createListView()),
                new Tab("Table", createTableView()),
                new Tab("Dummy")
                );

        BorderPane root = new BorderPane(tabPane);
        root.setBottom(new HBox(10, new Button("dummy for focus")));
        scene = new Scene(root, 300, 400);

        scene.getAccelerators().put(new KeyCodeCombination(KeyCode.F1), this::editNext);
        scene.getAccelerators().put(new KeyCodeCombination(KeyCode.F1, KeyCombination.CONTROL_DOWN), this::editPrevious);
        scene.getAccelerators().put(new KeyCodeCombination(KeyCode.F3), this::focusNextCell);
        scene.getAccelerators().put(new KeyCodeCombination(KeyCode.F4), this::selectNextCell);

        // quick check: implement stopEdit on focus lost to external control
        scene.focusOwnerProperty().addListener((src, ov, nv) -> {
//            String oldOwner = ov != null ? ov.getClass().getName() : "null";
//            String newOwner = nv != null ? nv.getClass().getName() : "null";
//            System.out.println("focusOwner: " + oldOwner + " / " + newOwner);
            // FIXME: something wrong here - StackOverflow if combined with editNext
            // was: incorrect childOf if focusOwner == listView
            if (childOf(ov, listView) && !(childOf(nv, listView))) {
                listView.stopEdit();
            }
        });
        root.setTop(new FlowPane(10, 10,
                new Label("F1 - editNext"),
                new Label("ctrl-F1 - editPrevious"),
                new Label("F3 - focusNextCell"),
                new Label("F4 - selectNextCell"),
                new Label()
                ));
        primaryStage.setScene(scene);
        primaryStage.setX(10);
        primaryStage.show();
    }

    private boolean childOf(Node child, Parent parent) {
        if (child == null || child == parent) return true;
        Parent current = child.getParent();
        while (current != null) {
            if (current == parent) return true;
            current = current.getParent();
        }
        return false;
    }

    private void selectNextCell() {
//        if (editNextOnCommit) return;
        Node content = getSelectedTabContent();
        if (content instanceof ListView) {
            ListView<?> listView = (ListView<?>) content;
            listView.getSelectionModel().selectNext();
        }
    }

    private void focusNextCell() {
//        if (editNextOnCommit) return;
        Node content = getSelectedTabContent();
        if (content instanceof ListView) {
            ListView<?> listView = (ListView<?>) content;
            listView.getFocusModel().focusNext();
        }

    }

    private void editNext() {
        Node content = getSelectedTabContent();
        if (editNextOnCommit) return;
        if (content instanceof ListView) {
            ListView<?> listView = (ListView) content;
            int editingIndex = listView.getEditingIndex();
            if (moveSelectionBefore) listView.getSelectionModel().selectNext();
            listView.edit(Math.min(editingIndex +1, listView.getItems().size() -1));
            if (moveSelectionAfter) listView.getSelectionModel().selectNext();
        }
    }

    private void editPrevious() {
        Node content = getSelectedTabContent();
        if (editNextOnCommit) return;
        if (content instanceof ListView) {
            ListView<?> listView = (ListView) content;
            int editingIndex = listView.getEditingIndex();
            if (moveSelectionBefore) listView.getSelectionModel().selectPrevious();
            listView.edit(Math.max(0, editingIndex - 1));
            if (moveSelectionAfter) listView.getSelectionModel().selectPrevious();
        }
    }
    // access virtualized control
    private Node getSelectedTabContent() {
        if (tabPane == null) {
            Node root = scene.getRoot();
            if (root instanceof BorderPane) {
                return ((BorderPane) root).getCenter();
            }
            return null;
        }
        Tab selected = tabPane.getSelectionModel().getSelectedItem();
        Node content = selected.getContent();
        if (content instanceof BorderPane) {
            content = ((BorderPane) content).getCenter();
        }
        return content;
    }

    private BorderPane createListView() {
        ListView<String> simpleList = new ListView<>(FXCollections
                .observableArrayList("Item1", "Item2", "Item3", "Item4")) {

                    @Override
                    public void stopEdit() {
//                        System.out.println("in stop: " + getEditingIndex() + " " + getEditState());
//                        new RuntimeException("who? ").printStackTrace();
                        super.stopEdit();
                    }

                    @Override
                    public void commitEdit() {
                        EditState<Integer> oldState = getEditState();
                        int editingIndex = oldState.getLocation();
//                        int editingIndex = getEditingIndex();
//                        // doesn't work - already stopped
//                        System.out.println("editingIndex on commit? " + editingIndex + " " + getEditState());
//                        new RuntimeException("who? ").printStackTrace();
                        super.commitEdit();
                        int selectedIndex = getSelectionModel().getSelectedIndex();
                        if (selectedIndex > -1 && editingIndex < getItems().size() -1 && editNextOnCommit) {
                            // FIXME not yet working - skin resets edit on save from commit handler
                            // fixed: commented skin handler
                            // FIXME: still doesn't work, even without interference from skin
                            // was: incorrect focusOwner listener
//                            System.out.println("ater super commit, trying to edit next " + (editingIndex) +
//                                    " old state: " + oldState + " state after commit " + getEditState());
                            // need to manually move selection - now: cell is focused, not the textField
                            getSelectionModel().selectNext();
//                            edit(editingIndex+1);
//                            System.out.println("selected: " + selectedIndex);
                            if (selectedIndex > -1) {
                                startEdit(new EditState<Integer>(new IndexedLocation(selectedIndex +1), EditAction.START));
                            }

                        } else if (editingIndex == getItems().size() - 1 && appendItemAndEditOnCommit) {
                            int size = getItems().size();
                            getItems().add("added at " + size);
                            getSelectionModel().select(size);
                            startEdit(new EditState<Integer>(new IndexedLocation(size), EditAction.START));
                        }

                    }


        };
        simpleList.setCellFactory(TextFieldListCell.forListView());
        simpleList.setEditable(true);
        simpleList.setDefaultStopEditAction(EditAction.COMMIT);

        simpleList.editStateProperty().addListener((src, ov, nv) -> {
            System.out.println("edit state change: " + ov + " " + nv);
        });
        simpleList.addEventHandler(ListView.editAnyEvent(), t -> {
//            System.out.println(t.getEventType() + " on " + t.getIndex());
            if (t.getEventType() == ListView.editCommitEvent()) {

            }
            if (t.getEventType() == ListView.editCancelEvent()) {
//                new RuntimeException("who? ").printStackTrace();
            }
        });
        // Note: with the new API, responsibilities are clearly separated
        // handler: saves edited value, must not touch editing state
        // listView edit methods (f.i. commit) trigger editing next/move selection/focus
        listView = simpleList;
        BorderPane tabContent = new BorderPane(simpleList);
        tabContent.setBottom(new HBox(10, new TextField("focus target on Tab")));
        return tabContent;
    }


    /**
     * For tableView we need an extractor on the property to see the cancel.
     */
    private TableView<MenuItem> createTableView() {
        ObservableList<MenuItem> itemsWithExtractor = FXCollections.observableArrayList(
                p -> new Observable[] {p.textProperty()});
        itemsWithExtractor.addAll(new MenuItem("first"), new MenuItem("second"), new MenuItem("third"));

        TableView<MenuItem> table = new TableView<>(itemsWithExtractor);
        table.setEditable(true);
        TableColumn<MenuItem, String> name = new TableColumn<>("Last Name");
        name.setCellFactory(TextFieldTableCell.forTableColumn());
        name.setCellValueFactory(cc -> cc.getValue().textProperty());
        TableColumn<MenuItem, String> style = new TableColumn<>("Last Name");
        style.setCellFactory(TextFieldTableCell.forTableColumn());
        style.setCellValueFactory(cc -> cc.getValue().styleProperty());

        table.getColumns().addAll(name, style);

        name.addEventHandler(TableColumn.editAnyEvent(), t ->
            System.out.println(t.getEventType() + " on " + t.getTablePosition().getRow()));

        return table;
    }

    public static void main(String[] args) {
        launch(args);
    }
}