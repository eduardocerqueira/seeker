//date: 2023-07-25T16:58:50Z
//url: https://api.github.com/gists/5d0809bcf06f22ebb833f2fb2b4d201a
//owner: https://api.github.com/users/meltoid872

package DataBase;

import org.bukkit.plugin.java.JavaPlugin;
import org.example.w.warp.DataBase.DataBase;

import javax.xml.crypto.Data;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;

import static java.sql.DriverManager.getConnection;

public class StartTracker extends JavaPlugin {
    @Override
    public void onEnable() {

        try{
            DataBase db = new DataBase();
            db.initializeDatabase();
        }catch (SQLException ex){
            System.out.println("Unable to connect do DataBase");

            ex.printStackTrace();
        }
    }
}