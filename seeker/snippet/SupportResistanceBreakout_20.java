//date: 2025-05-05T16:54:34Z
//url: https://api.github.com/gists/50eeccd48f70c7185dfef1700fda2780
//owner: https://api.github.com/users/vonWerra

import com.dukascopy.api.*;
import com.dukascopy.api.feed.*;
import com.dukascopy.api.feed.util.TimePeriodAggregationFeedDescriptor;
import com.dukascopy.api.indicators.*;
import javax.swing.*;
import javax.swing.text.*;
import java.awt.*;
import java.io.*;
import java.time.format.DateTimeFormatter;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;

public class SupportResistanceBreakout_20 implements IStrategy {
    private IEngine engine;
    private IConsole console;
    private IHistory history;
    private IContext context;
    private IIndicators indicators;

    @Configurable("Instrumenty") public String instrumentsString = "EUR/USD,GBP/USD,USD/JPY";
    @Configurable("Perioda indikatoru S&R") public Period indicatorPeriod = Period.FIFTEEN_MINS;
    @Configurable("Perioda grafu") public Period chartPeriod = Period.FIVE_MINS;
    @Configurable("Riziko (%)") public double riskPercent = 1.0;
    @Configurable("Debug") public boolean enableDebug = true;
    @Configurable("Zvuk") public boolean enableSound = true;
    @Configurable("Stop Loss (pipy)") public int stopLossPips = 5;
    @Configurable("Take Profit násobitel") public double tpMultiplier = 2.0;
    @Configurable("Logování na disk") public boolean enableFileLogging = true;
    @Configurable("Povolit GUI") public boolean enableGui = true;

    private Set<Instrument> instruments = new HashSet<>();
    private DateTimeFormatter dateFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss").withZone(ZoneId.of("GMT"));
    private Map<Instrument, Integer> lastAlertIndices = new ConcurrentHashMap<>();
    private Map<Instrument, Map<Long, double[]>> cache = new ConcurrentHashMap<>();
    private Map<Instrument, Long> lastProcessedBar = new ConcurrentHashMap<>();
    private ConcurrentLinkedQueue<String> alertBuffer = new ConcurrentLinkedQueue<>();
    private JFrame alertFrame;
    private JTextPane alertTextPane;
    private StyledDocument alertDoc;
    private PrintWriter logWriter;
    private int totalAlerts = 0;
    private long lastMemoryLogTime = 0;
    private int barsProcessed = 0;
    private static final int BARS_TO_WAIT = 3;
    private static final int SHIFT = 2;
    private static final int MAX_CACHE_SIZE = 50;
    private static final long GUI_UPDATE_INTERVAL = 1000;
    private static final long MEMORY_LOG_INTERVAL = 600000;

    private Map<Instrument, Boolean> isLongActive = new ConcurrentHashMap<>();
    private Map<Instrument, Boolean> isShortActive = new ConcurrentHashMap<>();
    private Map<Instrument, String> lastAlerts = new ConcurrentHashMap<>();
    private boolean isAlertWindowOpen = false;
    private Map<Instrument, Boolean> autoTradingEnabled = new ConcurrentHashMap<>();
    private Map<Instrument, JButton[]> tradingButtons = new ConcurrentHashMap<>();
    private Map<Instrument, Boolean> trailingStopActive = new ConcurrentHashMap<>();
    private Map<Instrument, Boolean> activeLong = new ConcurrentHashMap<>();
    private Map<Instrument, Boolean> activeShort = new ConcurrentHashMap<>();

    @Override
    public void onStart(IContext context) throws JFException {
        this.engine = context.getEngine();
        this.console = context.getConsole();
        this.history = context.getHistory();
        this.context = context;
        this.indicators = context.getIndicators();

        instruments.clear();
        for (String instr : instrumentsString.split(",")) {
            try {
                Instrument instrument = Instrument.fromString(instr.trim().toUpperCase());
                instruments.add(instrument);
                cache.put(instrument, new LinkedHashMap<Long, double[]>() {
                    @Override
                    protected boolean removeEldestEntry(Map.Entry<Long, double[]> eldest) {
                        return size() > MAX_CACHE_SIZE;
                    }
                });
                lastProcessedBar.put(instrument, 0L);
                isLongActive.put(instrument, false);
                isShortActive.put(instrument, false);
                lastAlerts.put(instrument, "");
                autoTradingEnabled.put(instrument, false);
                trailingStopActive.put(instrument, false);
                activeLong.put(instrument, false);
                activeShort.put(instrument, false);
                logMessage("Přidán nástroj: " + instrument, false);
            } catch (IllegalArgumentException e) {
                logMessage("Neplatný instrument: " + instr + ", chyba: " + e.getMessage(), true);
            }
        }

        context.setSubscribedInstruments(instruments);
        logMessage("Přihlášeno " + instruments.size() + " nástrojů: " + instruments, false);

        if (indicators.getIndicator("S&R") == null) {
            logMessage("Indikátor S&R není dostupný!", true);
            context.stop();
            return;
        }

        if (enableFileLogging) {
            initLogFile();
        }

        logMessage("Strategie spuštěna v Local režimu pro " + instruments.size() + " nástrojů", false);
        if (enableGui) {
            initAlertWindow();
            startGuiUpdateThread();
        }
    }

    private void initAlertWindow() {
        SwingUtilities.invokeLater(() -> {
            if (alertFrame == null || !alertFrame.isVisible()) {
                alertFrame = new JFrame("Breakout Alerts");
                alertFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
                alertFrame.setSize(800, 600);
                alertFrame.setLayout(new BorderLayout());

                JPanel controlPanel = new JPanel();
                controlPanel.setLayout(new GridLayout(instruments.size(), 5));

                for (Instrument instrument : instruments) {
                    JLabel instrumentLabel = new JLabel(instrument.toString());
                    instrumentLabel.setForeground(Color.BLACK);
                    JButton longButton = new JButton("LONG");
                    JButton shortButton = new JButton("SHORT");
                    JButton beButton = new JButton("BE");
                    JButton tsButton = new JButton("TS");

                    beButton.setEnabled(false);
                    tsButton.setEnabled(false);

                    longButton.addActionListener(e -> toggleButton(instrument, true, longButton, shortButton));
                    shortButton.addActionListener(e -> toggleButton(instrument, false, longButton, shortButton));
                    beButton.addActionListener(e -> setBreakEven(instrument, beButton));
                    tsButton.addActionListener(e -> toggleTrailingStop(instrument, tsButton));

                    tradingButtons.put(instrument, new JButton[]{longButton, shortButton, beButton, tsButton});

                    controlPanel.add(instrumentLabel);
                    controlPanel.add(longButton);
                    controlPanel.add(shortButton);
                    controlPanel.add(beButton);
                    controlPanel.add(tsButton);
                }

                alertTextPane = new JTextPane();
                alertTextPane.setEditable(false);
                alertTextPane.setFont(new Font("Arial", Font.PLAIN, 14));
                alertDoc = alertTextPane.getStyledDocument();

                Style defaultStyle = alertDoc.addStyle("default", null);
                StyleConstants.setForeground(defaultStyle, Color.BLACK);

                Style highlightStyle = alertDoc.addStyle("highlight", null);
                StyleConstants.setForeground(highlightStyle, new Color(255, 127, 39));

                JScrollPane scrollPane = new JScrollPane(alertTextPane);
                alertFrame.add(scrollPane, BorderLayout.CENTER);
                alertFrame.add(controlPanel, BorderLayout.NORTH);

                alertFrame.setVisible(true);
                isAlertWindowOpen = true;

                alertFrame.addWindowListener(new java.awt.event.WindowAdapter() {
                    @Override
                    public void windowClosed(java.awt.event.WindowEvent windowEvent) {
                        isAlertWindowOpen = false;
                    }
                });

                SwingUtilities.invokeLater(() -> {
                    try {
                        checkOpenPositionsAtStart();
                    } catch (JFException e) {
                        logMessage("Chyba při kontrole otevřených pozic: " + e.getMessage(), true);
                    }
                });
            }
        });
    }

    private void checkOpenPositionsAtStart() throws JFException {
        for (Instrument instrument : instruments) {
            IOrder openOrder = getOpenOrder(instrument);  // Opravený překlep
            if (openOrder != null) {
                JButton[] buttons = tradingButtons.get(instrument);
                if (buttons != null) {
                    buttons[2].setEnabled(true); // BE
                    buttons[3].setEnabled(true); // TS
                }
            }
        }
    }

    private void initLogFile() {
        try {
            String timestamp = dateFormatter.format(ZonedDateTime.now()).replace(":", "-");
            String logFileName = "strategy_log_" + timestamp + ".txt";
            File logFile = new File(logFileName);
            logWriter = new PrintWriter(new FileWriter(logFile, true), true);
            logMessage("Logovací soubor: " + logFile.getAbsolutePath(), false);
        } catch (IOException e) {
            console.getErr().println("Chyba při vytváření logovacího souboru: " + e.getMessage());
        }
    }

    private void toggleButton(Instrument instrument, boolean isLong, JButton longButton, JButton shortButton) {
        if (isLong) {
            boolean currentState = activeLong.getOrDefault(instrument, false);
            activeLong.put(instrument, !currentState);
            if (!currentState) {
                longButton.setForeground(Color.GREEN.darker());
                shortButton.setForeground(Color.BLACK);
                activeShort.put(instrument, false);
                logMessage("Čekání na průraz nad Maximus pro LONG: " + instrument, false);
            } else {
                longButton.setForeground(Color.BLACK);
                logMessage("Deaktivováno čekání na LONG: " + instrument, false);
            }
        } else {
            boolean currentState = activeShort.getOrDefault(instrument, false);
            activeShort.put(instrument, !currentState);
            if (!currentState) {
                shortButton.setForeground(Color.GREEN.darker());
                longButton.setForeground(Color.BLACK);
                activeLong.put(instrument, false);
                logMessage("Čekání na průraz pod Minimus pro SHORT: " + instrument, false);
            } else {
                shortButton.setForeground(Color.BLACK);
                logMessage("Deaktivováno čekání na SHORT: " + instrument, false);
            }
        }
    }

    private void setBreakEven(Instrument instrument, JButton beButton) {
        logMessage("Tlačítko BE stisknuto pro: " + instrument, false);
        try {
            IOrder order = getOpenOrder(instrument);
            if (order != null) {
                double entryPrice = order.getOpenPrice();
                logMessage("Nastavuji SL na BE: " + entryPrice + " pro pozici: " + order.getLabel(), false);
                
                context.executeTask(new Callable<Void>() {
                    @Override
                    public Void call() throws Exception {
                        try {
                            order.setStopLossPrice(entryPrice);
                            logMessage("SL úspěšně posunut na BE pro: " + instrument, false);
                            SwingUtilities.invokeLater(() -> {
                                appendToAlertWindow(instrument, "SL posunut na BE", System.currentTimeMillis());
                                beButton.setForeground(Color.BLUE);
                            });
                        } catch (JFException e) {
                            logMessage("Chyba při posunu SL na BE: " + e.getMessage(), true);
                        }
                        return null;
                    }
                });
            } else {
                logMessage("Žádná otevřená pozice pro: " + instrument, false);
            }
        } catch (JFException e) {
            logMessage("Chyba při posunu SL na BE: " + e.getMessage(), true);
        }
    }

    private void toggleTrailingStop(Instrument instrument, JButton tsButton) {
        boolean currentState = trailingStopActive.getOrDefault(instrument, false);
        trailingStopActive.put(instrument, !currentState);
        if (!currentState) {
            tsButton.setForeground(Color.BLUE);
            logMessage("Trailing stop aktivován pro: " + instrument, false);
            appendToAlertWindow(instrument, "Trailing stop aktivován", System.currentTimeMillis());
        } else {
            tsButton.setForeground(Color.BLACK);
            logMessage("Trailing stop deaktivován pro: " + instrument, false);
            appendToAlertWindow(instrument, "Trailing stop deaktivován", System.currentTimeMillis());
        }
    }

    private void startGuiUpdateThread() {
        new Thread(() -> {
            while (!Thread.currentThread().isInterrupted()) {
                try {
                    Thread.sleep(GUI_UPDATE_INTERVAL);
                    updateAlertWindow();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }).start();
    }

    private void updateAlertWindow() {
        SwingUtilities.invokeLater(() -> {
            if (alertFrame == null || !alertFrame.isVisible()) {
                return;
            }

            while (!alertBuffer.isEmpty()) {
                String alertMessage = alertBuffer.poll();
                try {
                    String[] parts = alertMessage.split(" \\| ");
                    String instrumentStr = parts[1];
                    Instrument instrument = Instrument.fromString(instrumentStr);

                    if (lastAlertIndices.containsKey(instrument)) {
                        int startIndex = lastAlertIndices.get(instrument);
                        int endIndex = startIndex + lastAlerts.get(instrument).length();
                        alertDoc.setCharacterAttributes(startIndex, endIndex - startIndex, alertDoc.getStyle("default"), true);
                    }

                    int startIndex = alertDoc.getLength();
                    alertDoc.insertString(startIndex, alertMessage + "\n", alertDoc.getStyle("highlight"));
                    lastAlertIndices.put(instrument, startIndex);
                    lastAlerts.put(instrument, alertMessage);

                    alertTextPane.setCaretPosition(alertDoc.getLength());
                } catch (BadLocationException e) {
                    logMessage("Chyba při přidávání alertu: " + e.getMessage(), true);
                }
            }
        });
    }

    private void appendToAlertWindow(Instrument instrument, String message, long time) {
        ZonedDateTime zonedDateTime = ZonedDateTime.ofInstant(java.time.Instant.ofEpochMilli(time), ZoneId.of("GMT"));
        String alertMessage = dateFormatter.format(zonedDateTime) + " | " + instrument + " | " + message;
        alertBuffer.offer(alertMessage);
        totalAlerts++;
        logMessage("Nový alert: " + alertMessage + ", celkem alertů: " + totalAlerts, false);

        if (enableSound) {
            java.awt.Toolkit.getDefaultToolkit().beep();
        }
    }

    private void logMessage(String message, boolean isError) {
        ZonedDateTime now = ZonedDateTime.now(ZoneId.of("GMT"));
        String logEntry = "[" + dateFormatter.format(now) + "] | " + message;
        if (isError) {
            console.getErr().println(logEntry);
        } else {
            console.getOut().println(logEntry);
        }
        if (enableFileLogging && logWriter != null) {
            logWriter.println(logEntry);
        }
    }

    @Override
    public void onBar(Instrument instrument, Period period, IBar askBar, IBar bidBar) throws JFException {
        if (period != chartPeriod) {
            return;
        }

        long currentTime = bidBar.getTime();
        long barStartTime = currentTime - (currentTime % chartPeriod.getInterval());

        Long lastBar = lastProcessedBar.getOrDefault(instrument, 0L);
        if (barStartTime <= lastBar) {
            return;
        }

        if (!instruments.contains(instrument)) {
            logMessage("Přeskočeno: " + instrument + ", není v seznamu nástrojů", true);
            return;
        }

        processBar(instrument, barStartTime, bidBar);
    }

    private void processBar(Instrument instrument, long barStartTime, IBar bidBar) throws JFException {
        java.util.List<IBar> chartBars = history.getBars(instrument, chartPeriod, OfferSide.BID, Filter.NO_FILTER, 1, barStartTime, 0);
        if (chartBars.isEmpty()) {
            logMessage("Nedostatek dat pro: " + instrument + " na periodě " + chartPeriod, true);
            return;
        }

        IBar currentBar = chartBars.get(0);
        barsProcessed++;

        double[] levels = getSRLevels(instrument, indicatorPeriod, currentBar.getTime());
        if (levels == null) {
            logMessage("Nepodařilo se načíst S&R úrovně pro: " + instrument, true);
            return;
        }

        double maximums = levels[0];
        double minimums = levels[1];

        double closePrice = currentBar.getClose();
        boolean longCondition = closePrice > maximums;
        boolean shortCondition = closePrice < minimums;

        checkTradingConditions(instrument, longCondition, shortCondition, closePrice, maximums, minimums);

        lastProcessedBar.put(instrument, barStartTime);

        if (enableGui && !isAlertWindowOpen) {
            initAlertWindow();
        }

        manageTrailingStop(instrument, bidBar);
    }

    private void checkTradingConditions(Instrument instrument, boolean longCondition, boolean shortCondition, double closePrice, double maximums, double minimums) throws JFException {
        if (activeLong.getOrDefault(instrument, false) && longCondition) {
            openPosition(instrument, true, closePrice, maximums, minimums);
            activeLong.put(instrument, false);
            JButton[] buttons = tradingButtons.get(instrument);
            buttons[0].setForeground(Color.BLACK);
        }

        if (activeShort.getOrDefault(instrument, false) && shortCondition) {
            openPosition(instrument, false, closePrice, maximums, minimums);
            activeShort.put(instrument, false);
            JButton[] buttons = tradingButtons.get(instrument);
            buttons[1].setForeground(Color.BLACK);
        }
    }

    private void manageTrailingStop(Instrument instrument, IBar bidBar) throws JFException {
        if (trailingStopActive.getOrDefault(instrument, false)) {
            IOrder order = getOpenOrder(instrument);
            if (order != null) {
                double[] levelsTS = getSRLevels(instrument, indicatorPeriod, bidBar.getTime());
                if (levelsTS != null) {
                    double maximus = levelsTS[0];
                    double minimus = levelsTS[1];
                    double pipValue = instrument.getPipValue();
                    double newSL;

                    if (order.isLong()) {
                        newSL = minimus - stopLossPips * pipValue;
                        if (newSL > order.getStopLossPrice()) {
                            order.setStopLossPrice(newSL);
                            logMessage("SL posunut pro LONG pozici: " + newSL, false);
                        }
                    } else {
                        newSL = maximus + stopLossPips * pipValue;
                        if (newSL < order.getStopLossPrice()) {
                            order.setStopLossPrice(newSL);
                            logMessage("SL posunut pro SHORT pozici: " + newSL, false);
                        }
                    }
                }
            }
        }
    }

    private double[] getSRLevels(Instrument instrument, Period period, long time) throws JFException {
        long periodInterval = period.getInterval();
        long periodStartTime = time - (time % periodInterval);

        Map<Long, double[]> instrumentCache = cache.get(instrument);
        if (instrumentCache.containsKey(periodStartTime)) {
            return instrumentCache.get(periodStartTime);
        }

        IFeedDescriptor feed = new TimePeriodAggregationFeedDescriptor(instrument, period, OfferSide.BID, Filter.NO_FILTER);
        Object[] result = indicators.calculateIndicator(feed, new OfferSide[]{OfferSide.BID}, "S&R",
                new IIndicators.AppliedPrice[]{IIndicators.AppliedPrice.CLOSE}, new Object[]{}, SHIFT);

        if (result != null && result.length >= 2 && result[0] instanceof Double && result[1] instanceof Double) {
            double[] levels = new double[]{(Double) result[0], (Double) result[1]};
            instrumentCache.put(periodStartTime, levels);
            logMessage("S&R úrovně pro: " + instrument + ", perioda: " + period + ", čas: " + dateFormatter.format(ZonedDateTime.ofInstant(java.time.Instant.ofEpochMilli(time), ZoneId.of("GMT"))) +
                       ", Maximums=" + levels[0] + ", Minimums=" + levels[1], false);
            return levels;
        } else {
            logMessage("Neplatný výstup indikátoru S&R pro: " + instrument, true);
            return null;
        }
    }

    private void openPosition(Instrument instrument, boolean isLong, double entryPrice, double maximums, double minimums) throws JFException {
        double pipValue = instrument.getPipValue();
        double slPrice = calculateSL(isLong, maximums, minimums, pipValue);
        double slDistance = Math.abs(entryPrice - slPrice);
        if (slDistance < 2 * pipValue) {
            logMessage("SL příliš blízko: " + instrument, true);
            return;
        }
        double tpPrice = calculateTP(isLong, entryPrice, slPrice, tpMultiplier);
        if (Math.abs(entryPrice - tpPrice) < 2 * pipValue) {
            logMessage("TP příliš blízko: " + instrument, true);
            return;
        }
        double balance = context.getAccount().getBalance();
        double riskAmount = balance * (riskPercent / 100.0);
        double slDistanceInPips = slDistance / pipValue;
        double pipValueInUSD = pipValue * 100000;
        double amount = riskAmount / (slDistanceInPips * pipValueInUSD);
        long amountInUnits = Math.round(amount * 1000000);
        if (amountInUnits < 1000) {
            logMessage("Příliš malá pozice: " + amountInUnits + " jednotek", true);
            return;
        }
        IEngine.OrderCommand command = isLong ? IEngine.OrderCommand.BUY : IEngine.OrderCommand.SELL;
        submitOrder(instrument, command, amountInUnits / 1000000.0, entryPrice, slPrice, tpPrice);
        JButton[] buttons = tradingButtons.get(instrument);
        buttons[2].setEnabled(true); // BE
        buttons[3].setEnabled(true); // TS
    }

    private double calculateSL(boolean isLong, double maximums, double minimums, double pipValue) {
        return isLong ? minimums - stopLossPips * pipValue : maximums + stopLossPips * pipValue;
    }

    private double calculateTP(boolean isLong, double entryPrice, double slPrice, double tpMultiplier) {
        double slDistance = Math.abs(entryPrice - slPrice);
        return isLong ? entryPrice + tpMultiplier * slDistance : entryPrice - tpMultiplier * slDistance;
    }

    private void submitOrder(Instrument instrument, IEngine.OrderCommand command, double amount, double entryPrice, double slPrice, double tpPrice) throws JFException {
        String label = "SR_" + (command.isLong() ? "Long" : "Short") + "_" + instrument.toString().replace("/", "") + "_" + System.currentTimeMillis();
        engine.submitOrder(label, instrument, command, amount, 0, 0, slPrice, tpPrice);
        logMessage("Otevřen obchod: " + instrument + ", " + (command.isLong() ? "Long" : "Short") +
                ", Vstup=" + entryPrice + ", SL=" + slPrice + ", TP=" + tpPrice + ", Velikost=" + amount, false);
        appendToAlertWindow(instrument, "Pozice otevřena: " + (command.isLong() ? "LONG" : "SHORT"), System.currentTimeMillis());
    }

    private IOrder getOpenOrder(Instrument instrument) throws JFException {
        for (IOrder order : engine.getOrders(instrument)) {
            if (order.getState() == IOrder.State.FILLED || order.getState() == IOrder.State.OPENED) {
                logMessage("Nalezena otevřená pozice: " + order.getLabel() + " pro instrument: " + instrument, false);
                return order;
            }
        }
        logMessage("Žádná otevřená pozice pro instrument: " + instrument, false);
        return null;
    }

    @Override
    public void onMessage(IMessage message) throws JFException {
        if (message.getType() == IMessage.Type.ORDER_CLOSE_OK) {
            IOrder order = message.getOrder();
            Instrument instrument = order.getInstrument();
            trailingStopActive.put(instrument, false);
            JButton[] buttons = tradingButtons.get(instrument);
            if (buttons != null) {
                buttons[2].setEnabled(false); // BE
                buttons[2].setForeground(Color.BLACK);
                buttons[3].setEnabled(false); // TS
                buttons[3].setForeground(Color.BLACK);
            }
            logMessage("Pozice zavřena: " + instrument, false);
            appendToAlertWindow(instrument, "Pozice zavřena", System.currentTimeMillis());
        }
    }

    @Override
    public void onTick(Instrument instrument, ITick tick) throws JFException {}
    @Override
    public void onAccount(IAccount account) throws JFException {}

    @Override
    public void onStop() throws JFException {
        int openPositions = 0;
        for (IOrder order : engine.getOrders()) {
            if (order.getState() == IOrder.State.FILLED || order.getState() == IOrder.State.OPENED) {
                openPositions++;
            }
        }
        int maxCacheSize = cache.values().stream().mapToInt(Map::size).max().orElse(0);
        logMessage("Strategie ukončena. Celkem alertů: " + totalAlerts + ", otevřených pozic: " + openPositions +
                   ", max. velikost cache: " + maxCacheSize + " záznamů", false);

        if (enableFileLogging && logWriter != null) {
            logWriter.close();
        }
    }
}