//date: 2024-07-11T16:52:08Z
//url: https://api.github.com/gists/e16d9842d2ed420d0832a880641a965d
//owner: https://api.github.com/users/voidregreso

import java.util.ArrayList;
import java.util.Calendar;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class CalcSLM extends Thread {
    private double[] AudioData;
    private double[] AudioDataBn;
    private double[] AudioDataLowPass;
    private double[] AudioDataSampled;
    private double[] AudioDataSampledPipe;
    private double[] AudioDataW;
    private double[][] Bn_Filters;
    private int[] BufferBn;
    private int BufferSize;
    private double[] CoefWeightingSLM;
    private double[] LBn_Filter;
    private double[] L_Slm;
    private double[] L_SlmMax;
    private double[] L_SlmMin;
    private double[] LevelBF;
    private double[] LevelSlm;
    private double[] LevelSlmMax;
    private double[] LevelSlmMin;
    private Thread Phase1_Thread;
    private int SAMPLE_RATE;
    private ArrayList<double[]> SaveLBn_Filter;
    private ArrayList<double[]> SaveL_Slm;
    private ArrayList<Boolean> Save_overload;
    private double Tf;
    private int[] indLowPass;
    private boolean isCalculating;
    private boolean isStart;
    private double meanPerf;
    private int[] orderBn;
    private int orderBnMax;
    private double sensitivity;
    private boolean statReady;
    private Calendar t0_Date;
    private static final double[] TWeighting = {1.0d, 0.125d, 1.0d, 0.035d, 1.5d};
    private static final double[] AWeighting = {0.637605921940775d, -1.940217422347462d, 1.996000069128823d, -0.7221615386020247d, 0.02916470248461033d, -3.933847323141513E-4d, 1.652131944801045E-6d, 1.0d, -3.550748857713154d, 4.943978609188165d, -3.53015441273966d, 1.501577219173829d, -0.4338578237572062d, 0.06921004766942955d};
    private static final double[] CWeighting = {0.5422220786334342d, -1.296252058209947d, 0.9935212085773927d, -0.2682289334358095d, 0.03008915353445947d, -0.001368622564076657d, 1.721685882204256E-5d, 1.0d, -3.011599123196967d, 3.54393337432347d, -2.248441217721788d, 0.9681778734815181d, -0.3066016398682175d, 0.05453514142542087d};
    private static final double[][] LowPass_Filter = {new double[]{4.891921476299063E-4d, 0.001558712942757707d, 0.003350238749207854d, 0.005012408463830158d, 0.005731976022467833d, 0.005012408463830157d, 0.003350238749207854d, 0.001558712942757707d, 4.891921476299062E-4d, 1.0d, -4.643635213368573d, 10.61698225763713d, -15.12012474139609d, 14.49844180083628d, -9.516275430943946d, 4.159947555232781d, -1.105576882887453d, 0.1368549466917421d}, new double[]{2.42160288557058E-5d, -1.876847373837585E-4d, 6.420485962173066E-4d, -0.001266484471383131d, 0.001575809184486534d, -0.001266484471383131d, 6.420485962173066E-4d, -1.876847373837585E-4d, 2.42160288557058E-5d, 1.0d, -7.867587558439409d, 27.08841602770628d, -53.3100641358524d, 65.58970769582444d, -51.66094283071723d, 25.43832206749055d, -7.159709002407171d, 0.8818577364120841d}};
    private static final int[] orderB1 = {6, 6, 6, 6, 6, 6, 6, 6, 8};
    private static final double[][] B1_Filters = {new double[]{1.186808907920508E-4d, 0.0d, -3.560426723761523E-4d, 0.0d, 3.560426723761523E-4d, 0.0d, -1.186808907920508E-4d, 0.0d, 0.0d, 1.0d, -5.736720561981201d, 13.77312657265598d, -17.71344093094615d, 12.87054816043125d, -5.009573499062782d, 0.8160683512750361d, 0.0d, 0.0d}, new double[]{8.583990300664719E-4d, 0.0d, -0.002575197090199416d, 0.0d, 0.002575197090199416d, 0.0d, -8.583990300664719E-4d, 0.0d, 0.0d, 1.0d, -5.364585518836975d, 12.20957962613471d, -15.08141755377497d, 10.66187005129953d, -4.091247086987422d, 0.6662582471549228d, 0.0d, 0.0d}, new double[]{0.005733959336021465d, 0.0d, -0.0172018780080644d, 0.0d, 0.0172018780080644d, 0.0d, -0.005733959336021465d, 0.0d, 0.0d, 1.0d, -4.351979374885559d, 8.563496296075726d, -9.628727083802332d, 6.520965611831645d, -2.523748002161262d, 0.4428190358067736d, 0.0d, 0.0d}, new double[]{0.0338794161018783d, 0.0d, -0.1016382483056349d, 0.0d, 0.1016382483056349d, 0.0d, -0.0338794161018783d, 0.0d, 0.0d, 1.0d, -1.67687154293526d, 2.385273648341682d, -1.933224356495606d, 1.416329842942955d, -0.5478129902514082d, 0.1885499246096933d, 0.0d, 0.0d}, new double[]{1.154560921011158E-4d, 0.0d, -3.463682763033474E-4d, 0.0d, 3.463682763033474E-4d, 0.0d, -1.154560921011158E-4d, 0.0d, 0.0d, 1.0d, -5.739747285842896d, 13.78651129794173d, -17.7370659997152d, 12.89131543863709d, -5.018643048196507d, 0.8176372494634873d, 0.0d, 0.0d}, new double[]{8.357577994062452E-4d, 0.0d, -0.002507273398218736d, 0.0d, 0.002507273398218736d, 0.0d, -8.357577994062452E-4d, 0.0d, 0.0d, 1.0d, -5.372561573982239d, 12.24160225265874d, -15.13315663940789d, 10.70360360693644d, -4.107880713713312d, 0.6688264223231669d, 0.0d, 0.0d}, new double[]{0.005590466789926769d, 0.0d, -0.01677140036978031d, 0.0d, 0.01677140036978031d, 0.0d, -0.005590466789926769d, 0.0d, 0.0d, 1.0d, -4.373937487602234d, 8.634425601343864d, -9.72745758338127d, 6.592176697956385d, -2.549830587506132d, 0.4462880346405692d, 0.0d, 0.0d}, new double[]{0.03309486714290591d, 0.0d, -0.09928460142871774d, 0.0d, 0.09928460142871774d, 0.0d, -0.03309486714290591d, 0.0d, 0.0d, 1.0d, -1.731230748817325d, 2.462915791967629d, -2.02291715483654d, 1.466704249750132d, -0.5719930544512775d, 0.1917918830801162d, 0.0d, 0.0d}, new double[]{0.01073553644049072d, 0.0d, -0.0429421457619629d, 0.0d, 0.06441321864294436d, 0.0d, -0.0429421457619629d, 0.0d, 0.01073553644049072d, 1.0d, -2.335352343507111d, 4.027699429508178d, -4.467295165177466d, 4.102496696125923d, -2.645320442985966d, 1.400015277793566d, -0.4595816325049209d, 0.1163159043625799d}};
    private static final int[] orderB3 = {6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8};
    private static final double[][] B3_Filters = {new double[]{2.247531075955384E-6d, 0.0d, -6.74259322786615E-6d, 0.0d, 6.74259322786615E-6d, 0.0d, -2.247531075955384E-6d, 0.0d, 0.0d, 1.0d, -5.908185958862305d, 14.5831495269493d, -19.2483129456045d, 14.32844929968648d, -5.703612667735593d, 0.9485149359622779d, 0.0d, 0.0d}, new double[]{4.454284185062785E-6d, 0.0d, -1.336285255518836E-5d, 0.0d, 1.336285255518836E-5d, 0.0d, -4.454284185062785E-6d, 0.0d, 0.0d, 1.0d, -5.871894001960754d, 14.42721209706313d, -18.98454862126826d, 14.11070270751768d, -5.617084272777452d, 0.9356207363462664d, 0.0d, 0.0d}, new double[]{8.812644364217943E-6d, 0.0d, -2.643793309265383E-5d, 0.0d, 2.643793309265383E-5d, 0.0d, -8.812644364217943E-6d, 0.0d, 0.0d, 1.0d, -5.819034099578857d, 14.20438317484201d, -18.61519977140609d, 13.81317705686606d, -5.50292818176776d, 0.9196358437719515d, 0.0d, 0.0d}, new double[]{1.73982807553103E-5d, 0.0d, -5.219484226593089E-5d, 0.0d, 5.219484226593089E-5d, 0.0d, -1.73982807553103E-5d, 0.0d, 0.0d, 1.0d, -5.741295695304871d, 13.88373104480404d, -18.09480807401687d, 13.40404079819452d, -5.351432382856329d, 0.8998976841700376d, 0.0d, 0.0d}, new double[]{3.425711262592332E-5d, 0.0d, -1.0277133787777E-4d, 0.0d, 1.0277133787777E-4d, 0.0d, -3.425711262592332E-5d, 0.0d, 0.0d, 1.0d, -5.626070737838745d, 13.42071249583357d, -17.35992591322977d, 12.83952356749229d, -5.149365645023486d, 0.8756462280193738d, 0.0d, 0.0d}, new double[]{6.722960901370481E-5d, 0.0d, -2.016888270411144E-4d, 0.0d, 2.016888270411144E-4d, 0.0d, -6.722960901370481E-5d, 0.0d, 0.0d, 1.0d, -5.454339742660523d, 12.75332483134355d, -16.32590943314617d, 12.06185322797567d, -4.878952103408239d, 0.8460348265420987d, 0.0d, 0.0d}, new double[]{1.314002783813929E-4d, 0.0d, -3.942008351441787E-4d, 0.0d, 3.942008351441787E-4d, 0.0d, -1.314002783813929E-4d, 0.0d, 0.0d, 1.0d, -5.197742342948914d, 11.80065680747373d, -14.88910948266056d, 11.0006963140012d, -4.516967514955222d, 0.8101614870724939d, 0.0d, 0.0d}, new double[]{2.555337288123694E-4d, 0.0d, -7.666011864371083E-4d, 0.0d, 7.666011864371083E-4d, 0.0d, -2.555337288123694E-4d, 0.0d, 0.0d, 1.0d, -4.814867854118347d, 10.4703548538738d, -12.94353414845231d, 9.584461238380177d, -4.034580271982686d, 0.7671269172053471d, 0.0d, 0.0d}, new double[]{4.938881894279072E-4d, 0.0d, -0.001481664568283722d, 0.0d, 0.001481664568283722d, 0.0d, -4.938881894279072E-4d, 0.0d, 0.0d, 1.0d, -4.24744188785553d, 8.689575411579455d, -10.42652421992687d, 7.773995140410766d, -3.399347105028015d, 0.7161331571043735d, 0.0d, 0.0d}, new double[]{9.474716690347286E-4d, 0.0d, -0.002842415007104186d, 0.0d, 0.002842415007104186d, 0.0d, -9.474716690347286E-4d, 0.0d, 0.0d, 1.0d, -3.41880214214325d, 6.486257841196675d, -7.402982023928352d, 5.637895618659915d, -2.582069960277061d, 0.6566321328498096d, 0.0d, 0.0d}, new double[]{0.001801388710961576d, 0.0d, -0.005404166132884728d, 0.0d, 0.005404166132884728d, 0.0d, -0.001801388710961576d, 0.0d, 0.0d, 1.0d, -2.241030097007752d, 4.153248198352895d, -4.146587508476247d, 3.482636992008473d, -1.572979333731574d, 0.5885295013225172d, 0.0d, 0.0d}, new double[]{0.003388669643023482d, 0.0d, -0.01016600892907045d, 0.0d, 0.01016600892907045d, 0.0d, -0.003388669643023482d, 0.0d, 0.0d, 1.0d, -0.6450859904289246d, 2.476953038568834d, -1.028821979559386d, 1.988405959631875d, -0.4126526440600777d, 0.512435665485136d, 0.0d, 0.0d}, new double[]{2.185006467979256E-6d, 0.0d, -6.555019403937767E-6d, 0.0d, 6.555019403937767E-6d, 0.0d, -2.185006467979256E-6d, 0.0d, 0.0d, 1.0d, -5.90941321849823d, 14.58847471587961d, -19.2574184270465d, 14.33606663366243d, -5.706696117514818d, 0.9489884833621957d, 0.0d, 0.0d}, new double[]{4.330640939262228E-6d, 0.0d, -1.299192281778668E-5d, 0.0d, 1.299192281778668E-5d, 0.0d, -4.330640939262228E-6d, 0.0d, 0.0d, 1.0d, -5.873671054840088d, 14.43478497719779d, -18.99724314517697d, 14.12106719333461d, -5.621138628261475d, 0.9362088276512939d, 0.0d, 0.0d}, new double[]{8.56869148186766E-6d, 0.0d, -2.570607444560298E-5d, 0.0d, 2.570607444560298E-5d, 0.0d, -8.56869148186766E-6d, 0.0d, 0.0d, 1.0d, -5.821634292602539d, 14.21524359946336d, -18.63303373007432d, 13.82738450903581d, -5.508291517242435d, 0.9203635889828207d, 0.0d, 0.0d}, new double[]{1.691831217931189E-5d, 0.0d, -5.075493653793567E-5d, 0.0d, 5.075493653793567E-5d, 0.0d, -1.691831217931189E-5d, 0.0d, 0.0d, 1.0d, -5.745134353637695d, 13.89939452902026d, -18.11998100732892d, 13.42362074783273d, -5.358568356560257d, 0.900794528211528d, 0.0d, 0.0d}, new double[]{3.331609987025772E-5d, 0.0d, -9.994829961077316E-5d, 0.0d, 9.994829961077316E-5d, 0.0d, -3.331609987025772E-5d, 0.0d, 0.0d, 1.0d, -5.631776928901672d, 13.44333549744678d, -17.39545765995817d, 12.8665469242259d, -5.158901239475906d, 0.8767451193332473d, 0.0d, 0.0d}, new double[]{6.539271571118053E-5d, 0.0d, -1.961781471335416E-4d, 0.0d, 1.961781471335416E-4d, 0.0d, -6.539271571118053E-5d, 0.0d, 0.0d, 1.0d, -5.462858557701111d, 12.78584249658385d, -16.37571033639785d, 12.0989801152722d, -4.891723000549375d, 0.8473722003635806d, 0.0d, 0.0d}, new double[]{1.278338211668953E-4d, 0.0d, -3.835014635006857E-4d, 0.0d, 3.835014635006857E-4d, 0.0d, -1.278338211668953E-4d, 0.0d, 0.0d, 1.0d, -5.210472583770752d, 11.84673145256947d, -14.95768533730357d, 11.05097986535167d, -4.534044903986594d, 0.8117749908448659d, 0.0d, 0.0d}, new double[]{2.486547323161881E-4d, 0.0d, -7.459641969485643E-4d, 0.0d, 7.459641969485643E-4d, 0.0d, -2.486547323161881E-4d, 0.0d, 0.0d, 1.0d, -4.833824992179871d, 10.53374049040153d, -13.03483687577073d, 9.65054994003544d, -4.057247749772237d, 0.7690522689636592d, 0.0d, 0.0d}, new double[]{4.807263118290098E-4d, 0.0d, -0.00144217893548703d, 0.0d, 0.00144217893548703d, 0.0d, -4.807263118290098E-4d, 0.0d, 0.0d, 1.0d, -4.275396227836609d, 8.77207944698992d, -10.54130781077834d, 7.856072987563412d, -3.42894873851875d, 0.718399585039945d, 0.0d, 0.0d}, new double[]{9.225320882914392E-4d, 0.0d, -0.002767596264874317d, 0.0d, 0.002767596264874317d, 0.0d, -9.225320882914392E-4d, 0.0d, 0.0d, 1.0d, -3.459233403205872d, 6.582942436731614d, -7.534759836936225d, 5.729497179961133d, -2.619578017564804d, 0.6592547878122365d, 0.0d, 0.0d}, new double[]{0.001754676550589477d, 0.0d, -0.005264029651768433d, 0.0d, 0.005264029651768433d, 0.0d, -0.001754676550589477d, 0.0d, 0.0d, 1.0d, -2.297515153884888d, 4.243851158357998d, -4.280097096758611d, 3.564417506291639d, -1.61807231896086d, 0.5915002072154247d, 0.0d, 0.0d}, new double[]{0.003302357833713145d, 0.0d, -0.009907073501139434d, 0.0d, 0.009907073501139434d, 0.0d, -0.003302357833713145d, 0.0d, 0.0d, 1.0d, -0.7193038631230593d, 2.517093053687256d, -1.152646750327178d, 2.024601855606834d, -0.4621009175241297d, 0.515711909010813d, 0.0d, 0.0d}, new double[]{9.163981698758372E-4d, 0.0d, -0.002749194509627512d, 0.0d, 0.002749194509627512d, 0.0d, -9.163981698758372E-4d, 0.0d, 0.0d, 1.0d, -3.469249248504639d, 6.607064461673431d, -7.567635882985316d, 5.752380082997649d, -2.628903058590546d, 0.659908107063422d, 0.0d, 0.0d}, new double[]{0.001743184111712447d, 0.0d, -0.005229552335137342d, 0.0d, 0.005229552335137342d, 0.0d, -0.001743184111712447d, 0.0d, 0.0d, 1.0d, -2.311524033546448d, 4.266652970796883d, -4.313518143349423d, 3.585022561850007d, -1.629302779958307d, 0.5922407511197738d, 0.0d, 0.0d}, new double[]{0.003281114335270523d, 0.0d, -0.009843343005811567d, 0.0d, 0.009843343005811567d, 0.0d, -0.003281114335270523d, 0.0d, 0.0d, 1.0d, -0.7377489320933819d, 2.52764094448969d, -1.18364273332719d, 2.034078779999315d, -0.4744544473687313d, 0.5165292102769342d, 0.0d, 0.0d}, new double[]{0.001126491637756248d, 0.0d, -0.004505966551024993d, 0.0d, 0.00675894982653749d, 0.0d, -0.004505966551024993d, 0.0d, 0.001126491637756248d, 1.0d, 1.645266555249691d, 3.94309722642277d, 3.940206302206695d, 4.871859807643263d, 3.004812381014974d, 2.295940652475144d, 0.7248825420574899d, 0.3359411160777301d}, new double[]{0.00253383237425594d, 0.0d, -0.01013532949702376d, 0.0d, 0.01520299424553564d, 0.0d, -0.01013532949702376d, 0.0d, 0.00253383237425594d, 1.0d, 4.442366659641266d, 10.11584021855619d, 14.65311957326574d, 14.70720468941874d, 10.37912621362843d, 5.069909436670455d, 1.572990921433747d, 0.2515307456097444d}};
    private boolean isSLM = true;
    private boolean isOverload = false;
    private boolean pauseCalc = true;
    private boolean isRealTime = true;
    private int cpt = 0;
    private boolean isBnFilters = false;
    private boolean isExpW = true;
    private int FreqWselect = 1;
    private int BnFilter = 0;
    private double tW = 0.025d;
    private final Object mPauseLock = new Object();
    private boolean isPause = true;

    public CalcSLM(boolean z, int i, int i2, double d) {
        this.isStart = z;
        this.SAMPLE_RATE = i;
        this.BufferSize = i2;
        this.sensitivity = d;
        this.isCalculating = false;
        initBuffers();
    }

    /** OK! **/
    public void initBuffers() {
        pauseCalc = true;
        while (isCalculating) {
            try {
                Thread.sleep(10L);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        LevelSlm = new double[12];
        LevelSlmMax = new double[9];
        LevelSlmMin = new double[]{1.0E20d, 1.0E20d, 1.0E20d, 1.0E20d, 1.0E20d, 1.0E20d, 1.0E20d, 1.0E20d, 1.0E20d};
        L_Slm = new double[16];
        L_SlmMax = new double[9];
        L_SlmMin = new double[9];
        CoefWeightingSLM = new double[5];
        CoefWeightingSLM[1] = 1.0d;
        for (int i = 1; i < 5; i++) {
            CoefWeightingSLM[i] = Math.exp(((-1.0d) / SAMPLE_RATE) / TWeighting[i]);
        }
        isOverload = false;
        AudioData = new double[BufferSize];
        AudioDataW = new double[(BufferSize + 6) * 3];
        cpt = 0;
        statReady = false;
        orderBnMax = 0;
        orderBn = BnFilter == 1 ? orderB1.clone() : orderB3.clone();
        for (int ord : orderBn) {
            if (orderBnMax < ord) orderBnMax = ord;
        }
        indLowPass = BnFilter == 1 ? new int[]{4, 8} : new int[]{12, 24};
        BufferBn = BnFilter == 1 ? new int[]{BufferSize / 32, BufferSize / 32, BufferSize / 32,
                BufferSize / 32, BufferSize / 2, BufferSize / 2, BufferSize / 2, BufferSize / 2, BufferSize} :
                new int[]{BufferSize / 32, BufferSize / 32, BufferSize / 32, BufferSize / 32, BufferSize / 32, BufferSize / 32,
                        BufferSize / 32, BufferSize / 32, BufferSize / 32, BufferSize / 32, BufferSize / 32, BufferSize / 32,
                        BufferSize / 2, BufferSize / 2, BufferSize / 2, BufferSize / 2, BufferSize / 2, BufferSize / 2,
                        BufferSize / 2, BufferSize / 2, BufferSize / 2, BufferSize / 2, BufferSize / 2, BufferSize / 2,
                        BufferSize, BufferSize, BufferSize, BufferSize, BufferSize};
        Bn_Filters = BnFilter == 1 ? B1_Filters.clone() : B3_Filters.clone();
        AudioDataBn = BnFilter == 1 ? new double[((BufferSize / 32) * 4) + ((BufferSize / 2) * 4) + BufferSize + 24 + 24 + 8] :
                new double[((BufferSize / 32) * 12) + ((BufferSize / 2) * 12) + (BufferSize * 5) + 72 + 72 + 18 + 16];
        AudioDataLowPass = new double[(BufferSize + 8) * 3];
        AudioDataSampled = new double[(BufferSize / 2) + BufferSize + (BufferSize / 32) + (orderBnMax * 3)];
        AudioDataSampledPipe = new double[(BufferSize / 2) + BufferSize + (BufferSize / 32) + (orderBnMax * 3)];
        LevelBF = new double[orderBn.length + 1];
        LBn_Filter = new double[orderBn.length + 1];
        Tf = (1.0d / SAMPLE_RATE) * BufferSize;
        pauseCalc = false;
        SaveLBn_Filter = new ArrayList<>();
        SaveL_Slm = new ArrayList<>();
        t0_Date = Calendar.getInstance();
    }

    public void onPause() {
        synchronized (this.mPauseLock) {
            this.isPause = true;
        }
    }

    public void onResume() {
        synchronized (this.mPauseLock) {
            this.isPause = false;
            this.mPauseLock.notify();
        }
    }

    public void onStop() {
        this.isStart = false;
    }

    /** OK! **/
    public void global_filters() {
        double d = 0.0d;
        for (int i : this.orderBn) {
            d += this.LevelBF[i];
        }
        this.LBn_Filter[this.orderBn.length] = Math.log10(d) * 10.0d;
        if (this.isExpW) {
            return;
        }
        this.LBn_Filter[this.orderBn.length] -= Math.log10(this.Tf * this.cpt) * 10.0d;
    }

    /** OK! **/
    @Override
    public void run() {
        double[] dArr = new double[25];
        while (this.isStart) {
            synchronized (this.mPauseLock) {
                while (this.isPause) {
                    try {
                        this.mPauseLock.wait();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
            this.isPause = true;
            long nanoTime = System.nanoTime();
            if (!this.pauseCalc) {
                this.isCalculating = true;
                if (this.isBnFilters) {
                    BandFilter();
                } else {
                    FreqWeightings();
                    IntegrateSLM();
                }
                if (this.cpt > 0) {
                    this.statReady = true;
                    saveData();
                }
                this.cpt++;
            } else {
                this.isCalculating = false;
            }
            long nanoTime2 = System.nanoTime();
            double d = 0.0d;
            for (int i = 0; i < 24; i++) {
                dArr[i] = dArr[i + 1];
                d += dArr[i];
            }
            dArr[24] = (nanoTime2 - nanoTime) * 1.0E-6d;
            this.meanPerf = (((d + dArr[24]) / 25.0d) / (((1.0d / this.SAMPLE_RATE) * this.BufferSize) * 1000.0d)) * 100.0d;
            this.isRealTime = this.meanPerf <= 100.0d;
        }
    }

    public void saveData() {
        if (this.isBnFilters) {
            this.SaveLBn_Filter.add(this.LBn_Filter.clone());
        } else {
            this.SaveL_Slm.add(this.L_Slm.clone());
        }
    }

    public void setAudioData(double[] dArr) {
        this.AudioData = dArr;
    }

    /** OK! **/
    public void FreqWeightings() {
        for (int i = 0; i < 6; i++) {
            AudioDataW[i] = AudioDataW[i + BufferSize];
            AudioDataW[BufferSize + 6 + i] = AudioDataW[BufferSize + 6 + i + BufferSize];
            AudioDataW[(BufferSize + 6) * 2 + i] = AudioDataW[(BufferSize + 6) * 2 + i + BufferSize];
        }
        int overloadCount = 0;
        for (int i = 6; i < BufferSize + 6; i++) {
            AudioDataW[i] = AudioData[i - 6] / sensitivity;
            if (AudioData[i - 6] > 32767.0 || AudioData[i - 6] < -32767.0) {
                overloadCount++;
                if (overloadCount > 3) {
                    isOverload = true;
                }
            } else {
                overloadCount = 0;
            }
            AudioDataW[i + BufferSize + 6] = AWeighting[0] * AudioDataW[i];
            AudioDataW[(BufferSize + 6) * 2 + i] = CWeighting[0] * AudioDataW[i];
            for (int j = 1; j <= 6; j++) {
                int index = i + BufferSize + 6;
                AudioDataW[index] += (AWeighting[j] * AudioDataW[i - j]) - (AWeighting[j + 6 + 1] * AudioDataW[index - j]);
                int cIndex = (BufferSize + 6) * 2 + i;
                AudioDataW[cIndex] += (CWeighting[j] * AudioDataW[i - j]) - (CWeighting[j + 6 + 1] * AudioDataW[cIndex - j]);
            }
        }
    }

    /** OK! **/
    public void IntegrateSLM() {
        double[] tempSum = new double[3];

        // First pass: calculate temporary sums
        for (int i = 0; i < BufferSize; i++) {
            for (int j = 0; j < 3; j++) {
                int baseIndex = ((j + 1) * 6 + BufferSize * j) + i;
                double energySum = calculateEnergy(AudioDataW[baseIndex - 1], AudioDataW[baseIndex]);

                if (cpt > 0) {
                    LevelSlm[j * 4] += energySum;
                }
                tempSum[j] += energySum;
            }
        }

        // Second pass: update levels
        for (int i = 0; i < BufferSize; i++) {
            for (int j = 0; j < 3; j++) {
                int baseIndex = ((j + 1) * 6 + BufferSize * j) + i;
                double energy = calculateEnergy(AudioDataW[baseIndex - 1], AudioDataW[baseIndex]);

                updateLevels(j, energy, tempSum[j]);
            }
        }

        // Final calculations
        calculateFinalLevels();
    }

    /** OK! **/
    private double calculateEnergy(double sample1, double sample2) {
        return ((1.0 / SAMPLE_RATE) * 2.5E9 * (sample1 * sample1 + sample2 * sample2)) / 2.0;
    }

    /** OK! **/
    private void updateLevels(int index, double energy, double tempSum) {
        for (int k = 1; k < TWeighting.length - 1; k++) {
            int levelIndex = index * 4 + k;
            double weightedEnergy = energy / TWeighting[k];

            if (k == 3) {
                double threshold = tempSum / Tf;
                if (threshold > LevelSlm[levelIndex]) {
                    LevelSlm[levelIndex] = LevelSlm[levelIndex] * CoefWeightingSLM[k] + weightedEnergy;
                } else {
                    LevelSlm[levelIndex] = LevelSlm[levelIndex] * CoefWeightingSLM[4] + weightedEnergy / TWeighting[4];
                }
            } else if (cpt > 0) {
                LevelSlm[levelIndex] = LevelSlm[levelIndex] * CoefWeightingSLM[k] + weightedEnergy;
            } else {
                LevelSlm[levelIndex] = LevelSlm[levelIndex] * CoefWeightingSLM[3] + energy / TWeighting[3];
            }

            updateMaxMinLevels(index, k, levelIndex);
        }
    }

    /** OK! **/
    private void updateMaxMinLevels(int index, int k, int levelIndex) {
        if (statReady) {
            int statIndex = (k - 1) + (index * 3);
            if (LevelSlmMax[statIndex] < LevelSlm[levelIndex]) {
                LevelSlmMax[statIndex] = LevelSlm[levelIndex];
            }
            if (LevelSlmMin[statIndex] > LevelSlm[levelIndex]) {
                LevelSlmMin[statIndex] = LevelSlm[levelIndex];
            }
        }
    }

    /** OK! **/
    private void calculateFinalLevels() {
        for (int i = 0; i < 3; i++) {
            int baseIndex = i * 5;
            if (cpt > 0) {
                L_Slm[baseIndex] = 10.0 * Math.log10(LevelSlm[i * 4]);
                L_Slm[baseIndex + 1] = L_Slm[baseIndex] - 10.0 * Math.log10(Tf * cpt);
            }
            for (int j = 2; j < 5; j++) {
                L_Slm[baseIndex + j] = 10.0 * Math.log10(LevelSlm[(j - 1) + (i * 4)]);
                int statIndex = (j - 2) + (i * 3);
                L_SlmMax[statIndex] = 10.0 * Math.log10(LevelSlmMax[statIndex]);
                L_SlmMin[statIndex] = 10.0 * Math.log10(LevelSlmMin[statIndex]);
            }
        }
        if (cpt > 0) {
            L_Slm[15] = L_Slm[6] + 10.0 * Math.log10((Tf * cpt) / 28800.0);
        }
    }

    public boolean getIsCalculating() {
        return this.isCalculating;
    }

    public void setIsSLM(boolean z) {
        this.isSLM = z;
    }

    public void setpauseCalc(boolean z) {
        this.pauseCalc = z;
    }

    public double getMeanPerf() {
        return this.meanPerf;
    }

    public double[] getL_Slm() {
        return this.L_Slm;
    }

    public double[] getL_SlmMax() {
        return this.L_SlmMax;
    }

    public double[] getL_SlmMin() {
        return this.L_SlmMin;
    }

    public void resetSLM() {
        initBuffers();
    }

    public boolean getisOverload() {
        return this.isOverload;
    }

    public double getSensitivity() {
        return this.sensitivity;
    }

    public void setSensitivity(double d) {
        this.sensitivity = d;
    }

    public void setisOverload(boolean z) {
        this.isOverload = z;
    }

    public boolean getiSRealTime() {
        return this.isRealTime;
    }

    public ArrayList<double[]> getSaveL_Slm() {
        return this.SaveL_Slm;
    }

    public ArrayList<double[]> getSaveLBn_Filter() {
        return this.SaveLBn_Filter;
    }

    public ArrayList<Boolean> getSave_overload() {
        return this.Save_overload;
    }

    public Calendar getT0_Date() {
        return this.t0_Date;
    }

    public int getCpt() {
        return this.cpt - 1;
    }

    public double getTf() {
        return this.Tf;
    }

    /** OK! **/
    public void BandFilter() {
        // Phase1 Thread
        this.Phase1_Thread = new Thread(new Runnable() {
            @Override
            public void run() {
                CalcSLM.this.FreqWeightings();
                CalcSLM.this.LowPassFilter(CalcSLM.this.FreqWselect);
                CalcSLM.this.SampledBuffer();
            }
        }, "Phase1 Thread");
        this.Phase1_Thread.start();

        // ExecutorService for BnFilters
        ExecutorService executorService1 = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        for (int i = 0; i < this.orderBn.length; i++) {
            executorService1.execute(new BnFilters(i));
        }
        executorService1.shutdown();

        // Wait for phase1Thread and executorService1 to complete
        try {
            this.Phase1_Thread.join();
            executorService1.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        // Copy AudioDataSampled to AudioDataSampledPipe
        System.arraycopy(this.AudioDataSampled, 0, this.AudioDataSampledPipe, 0, this.AudioDataSampled.length);

        // ExecutorService for IntegrateFilters
        ExecutorService executorService2 = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        for (int i = 0; i < this.orderBn.length; i++) {
            executorService2.execute(new IntegrateFilters(i));
        }
        executorService2.shutdown();

        // Wait for executorService2 to complete
        try {
            executorService2.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        // Final global filters
        global_filters();
    }

    /** OK! **/
    public void LowPassFilter(int n) {
        int bufferSize = this.BufferSize;
        for (int i = 0; i < 8; ++i) {
            AudioDataLowPass[i] = AudioDataLowPass[i + bufferSize];
            AudioDataLowPass[bufferSize + 8 + i] = AudioDataLowPass[bufferSize + 8 + i + bufferSize];
            AudioDataLowPass[(bufferSize + 8) * 2 + i] = AudioDataLowPass[(bufferSize + 8) * 2 + i + bufferSize];
        }
        for (int n2 = 8; n2 < bufferSize + 8; n2++) {
            AudioDataLowPass[n2] = AudioDataW[(bufferSize + 6) * n + n2 - 2];
            AudioDataLowPass[n2 + bufferSize + 8] = LowPass_Filter[0][0] * AudioDataLowPass[n2];
            AudioDataLowPass[(bufferSize + 8) * 2 + n2] = LowPass_Filter[1][0] * AudioDataLowPass[n2];
            for (int j = 1; j <= 8; j++) {
                int n3 = n2 + bufferSize + 8;
                int n10 = (bufferSize + 8) * 2 + n2;
                int n6 = n2 - j;
                int n9 = j + 1 + 8;
                AudioDataLowPass[n3] += LowPass_Filter[0][j] * AudioDataLowPass[n6] - LowPass_Filter[0][n9] * AudioDataLowPass[n3 - j];
                AudioDataLowPass[n10] += LowPass_Filter[1][j] * AudioDataLowPass[n6] - LowPass_Filter[1][n9] * AudioDataLowPass[n10 - j];
            }
        }
    }

    /** OK! **/
    public void SampledBuffer() {
        int halfBufferSize = this.BufferSize / 2;
        int oneThirtySecondBufferSize = this.BufferSize / 32;

        for (int i = 0; i < this.orderBnMax; i++) {
            double[] dArr = this.AudioDataSampled;
            int i4 = this.BufferSize;
            dArr[i] = dArr[i + i4];
            dArr[i4 + this.orderBnMax + i] = dArr[i4 + this.orderBnMax + i + halfBufferSize];
            dArr[i4 + this.orderBnMax + halfBufferSize + this.orderBnMax + i] = dArr[i4 + this.orderBnMax + halfBufferSize + this.orderBnMax + i + oneThirtySecondBufferSize];
        }

        System.arraycopy(this.AudioDataLowPass, 8, this.AudioDataSampled, this.orderBnMax, this.BufferSize);

        for (int i = 0; i < halfBufferSize; i++) {
            this.AudioDataSampled[i + this.BufferSize + (this.orderBnMax * 2)] = this.AudioDataLowPass[this.BufferSize + 16 + (i * 2)];
        }

        for (int i = 0; i < oneThirtySecondBufferSize; i++) {
            this.AudioDataSampled[i + this.BufferSize + (this.orderBnMax * 3) + halfBufferSize] = this.AudioDataLowPass[this.BufferSize + 24 + this.BufferSize + (i * 32)];
        }
    }

    /** OK! **/
    public class BnFilters implements Runnable {
        private int m;

        public BnFilters(int i) {
            this.m = i;
        }

        @Override
        public void run() {
            int totalOrderBn = 0;
            for (int i = 0; i < m; i++) {
                totalOrderBn += CalcSLM.this.BufferBn[i] + CalcSLM.this.orderBn[i];
            }

            for (int i = 0; i < CalcSLM.this.orderBn[m]; i++) {
                CalcSLM.this.AudioDataBn[i + totalOrderBn] = CalcSLM.this.AudioDataBn[CalcSLM.this.BufferBn[m] + i + totalOrderBn];
            }

            int bufferOffset = CalcSLM.this.BufferSize + CalcSLM.this.orderBnMax * 3 + CalcSLM.this.BufferSize / 2;
            int orderOffset = CalcSLM.this.orderBn[m] + totalOrderBn;

            if (m < CalcSLM.this.indLowPass[0]) {
                processAudioData(bufferOffset, orderOffset, 0);
            } else if (m < CalcSLM.this.indLowPass[1]) {
                processAudioData(CalcSLM.this.BufferSize + CalcSLM.this.orderBnMax * 2, orderOffset, 0);
            } else {
                processAudioData(CalcSLM.this.orderBnMax, orderOffset, 6);
            }
        }

        private void processAudioData(int bufferOffset, int orderOffset, int extraFilterCount) {
            double[] filters = CalcSLM.this.Bn_Filters[m];
            for (int i = 0; i < CalcSLM.this.BufferBn[m]; i++) {
                double result = calculateFilteredValue(filters, bufferOffset, orderOffset, i);

                if (extraFilterCount > 0) {
                    result += calculateExtraFilteredValue(filters, bufferOffset, orderOffset, i);
                }

                CalcSLM.this.AudioDataBn[orderOffset + i] = result;
            }
        }

        private double calculateFilteredValue(double[] filters, int bufferOffset, int orderOffset, int index) {
            return filters[0] * CalcSLM.this.AudioDataSampledPipe[bufferOffset + index]
                    - filters[10] * CalcSLM.this.AudioDataBn[orderOffset + index - 1]
                    + filters[2] * CalcSLM.this.AudioDataSampledPipe[bufferOffset + index - 2]
                    - filters[11] * CalcSLM.this.AudioDataBn[orderOffset + index - 2]
                    - filters[12] * CalcSLM.this.AudioDataBn[orderOffset + index - 3]
                    + filters[4] * CalcSLM.this.AudioDataSampledPipe[bufferOffset + index - 4]
                    - filters[13] * CalcSLM.this.AudioDataBn[orderOffset + index - 4]
                    - filters[14] * CalcSLM.this.AudioDataBn[orderOffset + index - 5]
                    + filters[6] * CalcSLM.this.AudioDataSampledPipe[bufferOffset + index - 6]
                    - filters[15] * CalcSLM.this.AudioDataBn[orderOffset + index - 6];
        }

        private double calculateExtraFilteredValue(double[] filters, int bufferOffset, int orderOffset, int index) {
            return -filters[16] * CalcSLM.this.AudioDataBn[orderOffset + index - 7]
                    + filters[8] * CalcSLM.this.AudioDataSampledPipe[bufferOffset + index - 8]
                    - filters[17] * CalcSLM.this.AudioDataBn[orderOffset + index - 8];
        }
    }

    /** OK! **/
    public class IntegrateFilters implements Runnable {
        private int m;

        public IntegrateFilters(int m) {
            this.m = m;
        }

        @Override
        public void run() {
            int n;
            double timeStep;
            if (m < indLowPass[0]) {
                n = 32;
                timeStep = 32.0 / SAMPLE_RATE;
            } else if (m < indLowPass[1]) {
                n = 2;
                timeStep = 2.0 / SAMPLE_RATE;
            } else {
                n = 1;
                timeStep = 1.0 / SAMPLE_RATE;
            }

            double exp, exp2;
            if (isExpW) {
                double negTimeStep = -timeStep;
                exp = Math.exp(negTimeStep / tW);
                exp2 = Math.exp(negTimeStep / 1.5);
            } else {
                tW = 1.0;
                exp = exp2 = 1.0;
            }

            int offset = 0;
            for (int i = 0; i < m; i++) {
                offset += BufferBn[i] + orderBn[i];
            }

            double integral = 0.0;
            if (tW == 0.035) {
                for (int j = 0; j < BufferSize / n; j++) {
                    int index = orderBn[m] + offset + j;
                    integral += 2.5E9 * timeStep * (Math.pow(AudioDataBn[index - 1], 2) + Math.pow(AudioDataBn[index], 2)) / 2.0;
                }
            }

            for (int k = 0; k < BufferSize / n; k++) {
                int index = orderBn[m] + offset + k;
                double squaredSum = Math.pow(AudioDataBn[index - 1], 2) + Math.pow(AudioDataBn[index], 2);
                if (tW == 0.035) {
                    if (integral / Tf > LevelBF[m]) {
                        LevelBF[m] = LevelBF[m] * exp + 1.0 / (tW * 4.0E-10) * timeStep * squaredSum / 2.0;
                    } else {
                        LevelBF[m] = LevelBF[m] * exp2 + 1.6666666666666665E9 * timeStep * squaredSum / 2.0;
                    }
                } else {
                    LevelBF[m] = LevelBF[m] * exp + 1.0 / (tW * 4.0E-10) * timeStep * squaredSum / 2.0;
                }
            }

            LBn_Filter[m] = 10.0 * Math.log10(LevelBF[m]);
            if (!isExpW) {
                LBn_Filter[m] -= 10.0 * Math.log10(Tf * cpt);
            }
        }
    }

    public void setIsBnFilters(boolean z) {
        this.isBnFilters = z;
    }

    public void setFreqWselect(int i) {
        this.FreqWselect = i;
    }

    public void settW(double d) {
        this.tW = d;
    }

    public void setisExpW(boolean z) {
        this.isExpW = z;
    }

    public void setBnFilter(int i) {
        this.BnFilter = i;
    }

    public int getBnFilter() {
        return this.BnFilter;
    }

    public double[] getLBn_Filters() {
        return this.LBn_Filter;
    }

    public int getNbrFilters() {
        return this.orderBn.length;
    }
}