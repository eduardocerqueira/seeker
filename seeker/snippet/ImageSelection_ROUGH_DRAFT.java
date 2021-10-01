//date: 2021-10-01T16:52:58Z
//url: https://api.github.com/gists/79c52597f7a3f3c1fa60340452292699
//owner: https://api.github.com/users/Alonso-del-Arte

package clipboardops;

import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.ClipboardOwner;
import java.awt.datatransfer.DataFlavor;
import java.awt.datatransfer.Transferable;
import java.awt.datatransfer.UnsupportedFlavorException;
import java.awt.Image;
import java.io.IOException;

public class ImageSelection implements Transferable, ClipboardOwner {
    
    private final Image heldImage;
    
    private static final DataFlavor SUPPORTED_FLAVOR = DataFlavor.imageFlavor;
    
    private boolean clipboardOwnershipFlag = false;

    // TODO: Write tests for this
    @Override
    public DataFlavor[] getTransferDataFlavors() {
        DataFlavor[] array = {DataFlavor.plainTextFlavor};
        return array;
    }

    // TODO: Write tests for this
    @Override
    public boolean isDataFlavorSupported(DataFlavor flavor) {
        return false;
    }

    // TODO: Write tests for this
    @Override
    public Object getTransferData(DataFlavor flavor) 
            throws UnsupportedFlavorException, IOException {
        return "Sorry, not implemented yet";
    }
    
    // TODO: Write tests for this
    @Override
    public void lostOwnership(Clipboard clipboard, Transferable contents) {
        // TODO: Write tests for this
    }
    
    // TODO: Write tests for this
    public boolean hasOwnership() {
        return this.clipboardOwnershipFlag;
    }

    public ImageSelection(Image image) {
        this.heldImage = image;
    }
    
}
