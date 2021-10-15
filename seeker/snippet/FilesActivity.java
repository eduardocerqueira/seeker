//date: 2021-10-15T16:56:20Z
//url: https://api.github.com/gists/5c437edf6a1f878a69e54d48382320b6
//owner: https://api.github.com/users/mitulvaghamshi

package package_name;

import android.app.*;
import android.os.*;
import android.view.*;
import android.widget.*;
import java.io.*;

public class FilesActivity extends Activity {
    private static final int READ_BLOCK_SIZE = 100;
	
	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);

		//---writing to files---
		try {
			FileOutputStream fOut = openFileOutput("textfile.txt", MODE_PRIVATE);
			OutputStreamWriter osw = new OutputStreamWriter(fOut);

			//---write the string to the file---
			osw.write("The quick brown fox jumps over the lazy dog");
			osw.close();

			//---display file saved message---
			Toast.makeText(getBaseContext(), "File saved successfully!", Toast.LENGTH_SHORT).show();
		} catch (IOException ioe) {
			ioe.printStackTrace();
		}

		//---reading from files---		
        try {  
            FileInputStream fIn = openFileInput("textfile.txt");
            InputStreamReader isr = new InputStreamReader(fIn);
            char[] inputBuffer = new char[READ_BLOCK_SIZE];
            String s = "";
            int charRead;
            while ((charRead = isr.read(inputBuffer)) > 0) {                    
                //---convert the chars to a String---
                String readString = String.copyValueOf(inputBuffer, 0, charRead);                    
                s += readString;
				inputBuffer = new char[READ_BLOCK_SIZE];
            } 
            isr.close();

            Toast.makeText(getBaseContext(), "File loaded successfully! " + s, Toast.LENGTH_SHORT).show();
        } catch (IOException ioe) { 
            ioe.printStackTrace(); 
        }		
	}

	private void method() {
		InputStream is = this.getResources().openRawResource(R.raw.textfile);
		BufferedReader br = new BufferedReader(new InputStreamReader(is));
		String str = null;
		try {
			while ((str = br.readLine()) != null) Toast.makeText(getBaseContext(), str, Toast.LENGTH_SHORT).show();
			is.close();
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void onClickSave(View view) {
		String str = "Hello!";
		try {
            //---SD Card Storage---
            File sdCard = Environment.getExternalStorageDirectory();
            File directory = new File (sdCard.getAbsolutePath() + "/MyFiles");
            directory.mkdirs();
            File file = new File(directory, "textfile.txt");
            FileOutputStream fOut = new FileOutputStream(file);

            /*			
			 FileOutputStream fOut =
			 openFileOutput("textfile.txt",
			 MODE_WORLD_READABLE);
			 */

			OutputStreamWriter osw = new OutputStreamWriter(fOut);

			//---write the string to the file---
			osw.write(str);
			osw.flush(); 
			osw.close();

			//---display file saved message---
			Toast.makeText(getBaseContext(), "File saved successfully!", Toast.LENGTH_SHORT).show();
		}
		catch (IOException ioe) {
			ioe.printStackTrace();
		}
	}

	public void onClickLoad(View view) {
		try {
			//---SD Storage---
            File sdCard = Environment.getExternalStorageDirectory();
            File directory = new File (sdCard.getAbsolutePath() + 
									   "/MyFiles");
            File file = new File(directory, "textfile.txt");
            FileInputStream fIn = new FileInputStream(file);
            InputStreamReader isr = new InputStreamReader(fIn);

            /*
			 FileInputStream fIn = 
			 openFileInput("textfile.txt");
			 InputStreamReader isr = new 
			 InputStreamReader(fIn);
			 */

			char[] inputBuffer = new char[READ_BLOCK_SIZE];
			String s = "";

			int charRead;
			while ((charRead = isr.read(inputBuffer))>0) {
				//---convert the chars to a String---
				String readString = String.copyValueOf(inputBuffer, 0, charRead);
				s += readString;
				inputBuffer = new char[READ_BLOCK_SIZE];
			}
			//---set the EditText to the text that has been 
			// read---
			Toast.makeText(getBaseContext(), "File loaded successfully!", Toast.LENGTH_SHORT).show();
		}
		catch (IOException ioe) {
			ioe.printStackTrace();
		}
	}
	
	private void onClickWrite() {
		//---writing to files---
        try { 
            if (IsExternalStorageAvailableAndWriteable()) {
                //---external Storage---
                File extStorage = getExternalFilesDir(null);
                File file = new File(extStorage, "textfile.txt");                                
                FileOutputStream fOut = new FileOutputStream(file);
                OutputStreamWriter osw = new OutputStreamWriter(fOut);  

                //---write the string to the file--- 
                osw.write("The quick brown fox jumps over the lazy dog");              
                osw.flush(); 
                osw.close();

                //---display file saved message---
                Toast.makeText(getBaseContext(), "File saved successfully!", Toast.LENGTH_SHORT).show();
            }            
        } 
        catch (IOException ioe) { 
            ioe.printStackTrace(); 
        }      
	}
	
	private void onClickRead() {
		//---reading from files---
        try {             
            if (IsExternalStorageAvailableAndWriteable()) {
                //---External Storage---                  
                File extStorage = getExternalFilesDir(null);
                File file = new File(extStorage, "textfile.txt");                                
                FileInputStream fIn = new FileInputStream(file);                
                InputStreamReader isr = new InputStreamReader(fIn); 
                char[] inputBuffer = new char[READ_BLOCK_SIZE];
                String s = "";
                int charRead;
                while ((charRead = isr.read(inputBuffer))>0) {                    
                    //---convert the chars to a String---
                    String readString = String.copyValueOf(inputBuffer, 0, charRead);                    
                    s += readString;
                    inputBuffer = new char[READ_BLOCK_SIZE];
                } 
                isr.close();
                Toast.makeText(getBaseContext(),"File loaded successfully! " + s, Toast.LENGTH_SHORT).show();                
            }
        } 
        catch (IOException ioe) { 
            ioe.printStackTrace(); 
        }
	}
	
	public boolean IsExternalStorageAvailableAndWriteable() {
        boolean externalStorageAvailable = false;
        boolean externalStorageWriteable = false;
        String state = Environment.getExternalStorageState();

        if (Environment.MEDIA_MOUNTED.equals(state)) {
            //---you can read and write the media---
            externalStorageAvailable = externalStorageWriteable = true;
        } else if (Environment.MEDIA_MOUNTED_READ_ONLY.equals(state)) {
            //---you can only read the media---
            externalStorageAvailable = true;
            externalStorageWriteable = false;
        } else {
            //---you cannot read nor write the media---
            externalStorageAvailable = externalStorageWriteable = false;
        }
        return externalStorageAvailable && externalStorageWriteable;
    }
	
	private void onClickReadCache() {
		//---Reading from the file in the Cache folder---
        try {             
            //---get the Cache directory---                  
            File cacheDir = getCacheDir();
            File file = new File(cacheDir, "textfile.txt");
            FileInputStream fIn = new FileInputStream(file);                
            InputStreamReader isr = new InputStreamReader(fIn); 
            char[] inputBuffer = new char[READ_BLOCK_SIZE];
            String s = "";
            int charRead;
            while ((charRead = isr.read(inputBuffer))>0) {                    
                //---convert the chars to a String---
                String readString = String.copyValueOf(inputBuffer, 0, charRead);                    
                s += readString;
				inputBuffer = new char[READ_BLOCK_SIZE];
            } 
            isr.close();
            Toast.makeText(getBaseContext(),"File loaded successfully! " + s, Toast.LENGTH_SHORT).show();
        } 
        catch (IOException ioe) { 
            ioe.printStackTrace(); 
        }
	}
	
	private void onClickWriteCache() {
		//---Saving to the file in the Cache folder---
        try { 
            //---get the Cache directory---
            File cacheDir = getCacheDir();            
            File file = new File(cacheDir.getAbsolutePath(), "textfile.txt");                                
            FileOutputStream fOut = new FileOutputStream(file);
            OutputStreamWriter osw = new OutputStreamWriter(fOut);  

            //---write the string to the file--- 
            osw.write("The quick brown fox jumps over the lazy dog");              
            osw.flush(); 
            osw.close();

            //---display file saved message---
            Toast.makeText(getBaseContext(), "File saved successfully!", Toast.LENGTH_SHORT).show();         
        } 
        catch (IOException ioe) { 
			ioe.printStackTrace(); 
        }    
	}
}
