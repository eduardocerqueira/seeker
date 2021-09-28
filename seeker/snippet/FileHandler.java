//date: 2021-09-28T17:09:36Z
//url: https://api.github.com/gists/32a90ee51e40671b9cb455910e5dfd57
//owner: https://api.github.com/users/mineheroesrpg

import java.io.File;

/**
 * Loads a file, and than you can get this file with the getFile() method:
 * Usage:
 * FileHandler fileHandler = new FileHandler(path);
 * File file = fileHandler.getFile();
 */

public class FileHandler {
    private final String path;
    private File file;

    public FileHandler(String path) {
        this.path = path;
        load();
    }

    private boolean load() {
        if (new File(getPath()).exists()) {
            this.file = new File(getPath());
            return true;
        } else {
            System.err.println(getPath() + " not exist!");
            return false;
        }
    }

    public String getPath() {
        return path;
    }

    public File getFile() {
        return file;
    }
}