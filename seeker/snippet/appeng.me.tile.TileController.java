//date: 2025-04-29T16:57:22Z
//url: https://api.github.com/gists/169235d6071918dfa2f65ed8702d505e
//owner: https://api.github.com/users/SrYthan

package appeng.me.tile;

import appeng.api.Blocks;
import appeng.api.IAEItemStack;
import appeng.api.IItemList;
import appeng.api.IWirelessTermHandler;
import appeng.api.Materials;
import appeng.api.TileRef;
import appeng.api.Util;
import appeng.api.WorldCoord;
import appeng.api.config.InterfaceCraftingMode;
import appeng.api.config.PowerUnits;
import appeng.api.config.ViewItems;
import appeng.api.events.GridErrorEvent;
import appeng.api.events.GridPatternUpdateEvent;
import appeng.api.events.GridStorageUpdateEvent;
import appeng.api.events.LocateableEventAnnounce;
import appeng.api.exceptions.AppEngTileMissingException;
import appeng.api.me.tiles.ICellContainer;
import appeng.api.me.tiles.IExtendedCellProvider;
import appeng.api.me.tiles.IGridMachine;
import appeng.api.me.tiles.IGridTileEntity;
import appeng.api.me.tiles.ILocateable;
import appeng.api.me.tiles.IOrientableTile;
import appeng.api.me.tiles.IPushable;
import appeng.api.me.tiles.IStorageAware;
import appeng.api.me.util.IAssemblerCluster;
import appeng.api.me.util.IAssemblerPattern;
import appeng.api.me.util.ICraftRequest;
import appeng.api.me.util.IGridCache;
import appeng.api.me.util.IGridInterface;
import appeng.api.me.util.IMEInventory;
import appeng.api.me.util.IMEInventoryHandler;
import appeng.common.AppEng;
import appeng.common.AppEngConfiguration;
import appeng.common.AppEngTextureRegistry;
import appeng.common.base.AppEngMultiBlock;
import appeng.common.grid.GridEnumeration;
import appeng.common.network.IAppEngNetworkTile;
import appeng.common.network.packets.PacketGridAnimate;
import appeng.common.registries.WirelessRangeResult;
import appeng.gui.AppEngGuiHandler;
import appeng.interfaces.INetworkNotifiable;
import appeng.interfaces.IPowerSharing;
import appeng.me.AssemblerPatternInventory;
import appeng.me.GridReference;
import appeng.me.MEInventoryHandler;
import appeng.me.MEInventoryNetwork;
import appeng.me.MEInventoryNull;
import appeng.me.METhrottle;
import appeng.me.basetiles.TileME;
import appeng.me.basetiles.TilePoweredBase;
import appeng.me.container.ContainerCraftingMonitor;
import appeng.me.container.ContainerTerminal;
import appeng.me.crafting.AssemblerCluster;
import appeng.me.crafting.CraftRequest;
import appeng.me.crafting.Crafting;
import appeng.me.crafting.CraftingInventory;
import appeng.me.crafting.CraftingJobPacket;
import appeng.me.crafting.CraftingManager;
import appeng.me.crafting.DelayedCraftRequest;
import appeng.me.crafting.ExternalCraftRequest;
import appeng.me.crafting.ICraftingManagerOwner;
import appeng.me.crafting.MissingMaterialsCraftRequest;
import appeng.me.crafting.PushCraftRequest;
import appeng.proxy.helpers.ILPInventory;
import appeng.util.AEItemStack;
import appeng.util.ItemList;
import appeng.util.ItemSorters;
import appeng.util.Platform;
import cpw.mods.fml.common.FMLLog;
import cpw.mods.fml.common.network.PacketDispatcher;
import cpw.mods.fml.common.network.Player;
import cpw.mods.fml.relauncher.Side;
import cpw.mods.fml.relauncher.SideOnly;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Date;
import java.util.Deque;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import net.minecraft.block.Block;
import net.minecraft.client.renderer.RenderBlocks;
import net.minecraft.client.renderer.Tessellator;
import net.minecraft.entity.Entity;
import net.minecraft.entity.EntityLiving;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.entity.player.EntityPlayerMP;
import net.minecraft.item.ItemStack;
import net.minecraft.nbt.NBTTagCompound;
import net.minecraft.network.packet.Packet;
import net.minecraft.network.packet.Packet250CustomPayload;
import net.minecraft.tileentity.TileEntity;
import net.minecraft.util.Icon;
import net.minecraft.world.IBlockAccess;
import net.minecraftforge.common.ForgeDirection;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.event.Event;
import net.minecraftforge.event.ForgeSubscribe;

public class TileController extends TilePoweredBase implements IOrientableTile, ILocateable, ICraftingManagerOwner, IExtendedCellProvider, IGridTileEntity, IGridInterface, IAppEngNetworkTile, INetworkNotifiable {
  public String EncryptionKey;
  
  public ForgeDirection orientation;
  
  public int gridIndex;
  
  private boolean sendUpdate;
  
  private boolean sendCraftingUpdate;
  
  private int TicksBetweenFlashes = 0;
  
  private boolean hasFlashed;
  
  private boolean enabled;
  
  int cables;
  
  List currentStatus;
  
  public float powerDrained = 0.0F;
  
  public float powerDrainedPersonal = 0.0F;
  
  boolean triggerUpdate;
  
  int oldFace;
  
  public float[] personalPowerUsageLog;
  
  public float[] realPowerUsageLog;
  
  public float realPowerUsageActiveTick;
  
  public float personalPowerUsageActiveTick;
  
  private METhrottle craftingThrottle;
  
  private METhrottle waitingThrottle;
  
  public ItemStack lastCraftingRequest;
  
  private List AllEntities;
  
  private List AllMachines;
  
  private IGridCache[] Caches;
  
  private List CellContainers;
  
  private List Assemblers;
  
  private List Interfaces;
  
  private List WirelessNodes;
  
  private List StorageAware;
  
  private List PowerSources;
  
  private List contentsViewingPlayers;
  
  private List craftingViewingPlayers;
  
  private List invChanges;
  
  private Deque CraftingQueue;
  
  private Deque WaitingQueue;
  
  private CraftingManager CManager;
  
  ItemList storedList;
  
  ItemList fullList;
  
  ItemList craftList;
  
  public float getPowerUsageAvg() {
    float avg = 0.0F;
    for (int x = 0; x < 20; x++)
      avg += this.realPowerUsageLog[x]; 
    return avg / 20.0F;
  }
  
  public float getPersonalPowerUsageAvg() {
    float avg = 0.0F;
    for (int x = 0; x < 20; x++)
      avg += this.personalPowerUsageLog[x]; 
    return avg / 20.0F;
  }
  
  private void pushPowerUsage() {
    for (int x = 1; x < 20; x++) {
      this.personalPowerUsageLog[x - 1] = this.personalPowerUsageLog[x];
      this.realPowerUsageLog[x - 1] = this.realPowerUsageLog[x];
    } 
    this.personalPowerUsageLog[19] = this.personalPowerUsageActiveTick;
    this.personalPowerUsageActiveTick = 0.0F;
    this.realPowerUsageLog[19] = this.realPowerUsageActiveTick;
    this.realPowerUsageActiveTick = 0.0F;
    this.maxStoredPower = getPowerUsageAvg() * 60.0F + 4000.0F;
  }
  
  public void OnCraftingChange(CraftingManager cm) {
    Iterator<EntityPlayer> ip = this.craftingViewingPlayers.iterator();
    while (ip.hasNext()) {
      EntityPlayer p = ip.next();
      if (p.openContainer != null) {
        if (p.openContainer instanceof ContainerCraftingMonitor) {
          ContainerCraftingMonitor ct = (ContainerCraftingMonitor)p.openContainer;
          ct.triggerUpdate();
          continue;
        } 
        ip.remove();
      } 
    } 
  }
  
  public void jobDone(CraftingManager cm) {
    if (cm == this.CManager)
      OnCraftingChange(cm); 
  }
  
  @ForgeSubscribe
  public void updatePatterns(GridPatternUpdateEvent pu) {
    if (pu.grid.getController() == this)
      this.cachedPatternSet = null; 
  }
  
  @ForgeSubscribe
  public void updateStorage(GridStorageUpdateEvent pu) {
    if (pu.grid != null && pu.grid.getController() == this)
      this.cachedCellArray = null; 
  }
  
  @ForgeSubscribe
  public void addToGrid(GridPatternUpdateEvent pu) {
    if (pu.grid != null && pu.grid.getController() == this)
      this.cachedPatternSet = null; 
  }
  
  public void init() {
    super.init();
    if (Platform.isClient())
      return; 
    MinecraftForge.EVENT_BUS.register(this);
    MinecraftForge.EVENT_BUS.post((Event)new LocateableEventAnnounce(this, LocateableEventAnnounce.LocateableEvent.Register));
  }
  
  protected void terminate() {
    super.terminate();
    if (Platform.isClient())
      return; 
    MinecraftForge.EVENT_BUS.post((Event)new LocateableEventAnnounce(this, LocateableEventAnnounce.LocateableEvent.Unregister));
    MinecraftForge.EVENT_BUS.unregister(this);
  }
  
  public void cancelJob(ItemStack job, ItemStack subjob) {
    List jlist = this.CManager.getPrereqs();
    List tl = new ArrayList();
    tl.addAll(0, jlist);
    for (CraftRequest cr : tl) {
      if (Platform.isSameItem(job, cr.getRequest())) {
        cr.cancel(subjob, this);
        OnCraftingChange((CraftingManager)null);
      } 
    } 
  }
  
  public List getJobList() {
    List<?> jobs = new ArrayList();
    List jlist = this.CManager.getPrereqs();
    for (CraftRequest cr : jlist) {
      if (cr.getAmount() > 0)
        Platform.sumItemToList(jobs, cr.getRequest()); 
    } 
    Collections.sort(jobs, ItemSorters.Accending_SortByID_Vanilla);
    return jobs;
  }
  
  public CraftingJobPacket getJobStatus(ItemStack is) {
    List jobs = this.CManager.getPrereqs();
    CraftingJobPacket cjp = new CraftingJobPacket();
    if (is != null) {
      cjp.Target = is;
      cjp.Target.stackSize = 0;
      for (CraftRequest cr : jobs) {
        if (Platform.isSameItemType(is, cr.getRequest())) {
          cjp.Target.stackSize += cr.getAmount();
          cr.populateJobPacket(cjp);
        } 
      } 
    } 
    return cjp;
  }
  
  public synchronized void resetWaitingQueue() {
    while (this.WaitingQueue.size() > 0) {
      CraftRequest cr = this.WaitingQueue.pop();
      while (cr.getAmount() > 0)
        cr.markCrafted(); 
    } 
    OnCraftingChange((CraftingManager)null);
  }
  
  public void printCraftingStatus() {
    AppEng.log(getName() + " is waiting on " + this.WaitingQueue.size() + " jobs");
    for (CraftRequest cr : this.WaitingQueue)
      cr.printJobDetails(); 
    AppEng.log(getName() + " is working on " + this.CraftingQueue.size() + " jobs");
    for (CraftRequest cr : this.CraftingQueue)
      cr.printJobDetails(); 
  }
  
  public boolean[] noScreen() {
    int rotation = getAERotationFromDirection(this.orientation);
    this.powerBar = false;
    return null;
  }
  
  public boolean[] screenOnly() {
    int rotation = getAERotationFromDirection(this.orientation);
    this.powerBar = true;
    return new boolean[] { (rotation != 5), (rotation != 4), (rotation != 2), (rotation != 0), (rotation != 1), (rotation != 3) };
  }
  
  @SideOnly(Side.CLIENT)
  public boolean renderWorldBlock(IBlockAccess world, int x, int y, int z, Block block, int modelId, RenderBlocks renderer) {
    renderer.setRenderBounds(0.0D, 0.0D, 0.0D, 1.0D, 1.0D, 1.0D);
    if (AppEngConfiguration.requirePower) {
      AppEngMultiBlock b = (AppEngMultiBlock)block;
      b.dontrender = noScreen();
      renderer.renderStandardBlock(block, x, y, z);
      b.dontrender = screenOnly();
      int bn = 15;
      Tessellator.instance.setColorOpaque_F(1.0F, 1.0F, 1.0F);
      Tessellator.instance.setBrightness(bn << 20 | bn << 4);
      renderFace(block, renderer, this.orientation);
      b.dontrender = null;
    } else {
      renderer.renderStandardBlock(block, x, y, z);
    } 
    return true;
  }
  
  public String getMsg() {
    String Message = "";
    if (this.enabled && this.currentStatus != null && this.currentStatus.size() > 0) {
      String Units = "{Units / t}";
      float conversionRate = 1.0F;
      DecimalFormat df = new DecimalFormat("0");
      switch (AppEngConfiguration.defaultUnits) {
        case EU:
          conversionRate = 0.5F;
          Units = "eu/t";
          df = new DecimalFormat("0.#");
          break;
        case MJ:
          conversionRate = 0.2F;
          Units = "mj/t";
          df = new DecimalFormat("0.#");
          break;
        case UE:
          conversionRate = 20.0F;
          Units = "j/t";
          break;
      } 
      float real = this.powerDrained * conversionRate;
      float personal = this.powerDrainedPersonal * conversionRate;
      String Extra = (real > personal) ? (" + " + df.format((real - personal))) : "";
      if (((TileME)this).hasPower) {
        Message = " - {Online}\n{Energy Used}: " + df.format(personal) + Extra + " " + Units;
      } else {
        Message = " - {Offline}\n{Power is low} ( " + df.format(personal) + Extra + " " + Units + " )";
      } 
    } else {
      Message = " - {Offline}\n{Controller Conflict}";
    } 
    return Message;
  }
  
  public ICraftRequest pushRequest(ItemStack willAdd, IPushable out, boolean allowCrafting) {
    this.craftingThrottle.wakeUp();
    return pushRequest(getName(), willAdd, out, allowCrafting, false);
  }
  
  public ICraftRequest pushRequest(String name, ItemStack willAdd, IPushable out, boolean allowCrafting) {
    this.craftingThrottle.wakeUp();
    return pushRequest(getName() + ((name == null) ? "" : ("-" + name)), willAdd, out, allowCrafting, false);
  }
  
  public ICraftRequest pushRequest(String name, ItemStack willAdd, IPushable out, boolean allowCrafting, boolean showInManager) {
    if (willAdd == null)
      return null; 
    PushCraftRequest pushCraftRequest = new PushCraftRequest(getName() + ((name == null) ? "" : ("-" + name)), willAdd, out, allowCrafting);
    this.craftingThrottle.wakeUp();
    this.CraftingQueue.add(pushCraftRequest);
    if (showInManager)
      this.CManager.requestedPreReqs((ICraftRequest)pushCraftRequest); 
    return (ICraftRequest)pushCraftRequest;
  }
  
  public ICraftRequest waitingRequest(ItemStack what) {
    if (what == null)
      return null; 
    this.craftingThrottle.hasAccomplishedWork();
    DelayedCraftRequest delayedCraftRequest = new DelayedCraftRequest(getName(), what);
    this.WaitingQueue.add(delayedCraftRequest);
    OnCraftingChange((CraftingManager)null);
    return (ICraftRequest)delayedCraftRequest;
  }
  
  boolean canLogisticsMake(ItemStack what) {
    if (what == null)
      return false; 
    try {
      for (TileRef ar : this.Interfaces) {
        TileInterfaceBase a = (TileInterfaceBase)ar.getTile();
        for (IMEInventory i : a.getLogisticsInv()) {
          if (i.containsItemType((IAEItemStack)AEItemStack.create(what)))
            return true; 
        } 
      } 
    } catch (AppEngTileMissingException e) {}
    return false;
  }
  
  boolean logisticsRequest(ItemStack what) {
    if (what == null)
      return false; 
    try {
      for (TileRef ar : this.Interfaces) {
        TileInterfaceBase a = (TileInterfaceBase)ar.getTile();
        for (IMEInventory i : a.getLogisticsInv()) {
          ILPInventory lpinv = (ILPInventory)i;
          List result = lpinv.requestCrafting(what);
          if (result == null || result.size() == 0)
            return true; 
        } 
      } 
    } catch (AppEngTileMissingException e) {}
    return false;
  }
  
  public ICraftRequest craftingRequest(ItemStack what, boolean showInManager, boolean recursive) {
    if (what == null)
      return null; 
    this.craftingThrottle.hasAccomplishedWork();
    if (recursive) {
      if (logisticsRequest(what)) {
        ExternalCraftRequest externalCraftRequest = new ExternalCraftRequest(getName(), what);
        externalCraftRequest.disablePreReqs();
        this.WaitingQueue.add(externalCraftRequest);
        return (ICraftRequest)externalCraftRequest;
      } 
      CraftRequest craftRequest = new CraftRequest(getName(), what);
      if (showInManager)
        this.CManager.requestedPreReqs((ICraftRequest)craftRequest); 
      this.CraftingQueue.add(craftRequest);
      OnCraftingChange((CraftingManager)null);
      return (ICraftRequest)craftRequest;
    } 
    CraftRequest ct = new CraftRequest(getName(), what);
    ct.allowPrereqs = false;
    if (showInManager)
      this.CManager.requestedPreReqs((ICraftRequest)ct); 
    this.CraftingQueue.add(ct);
    OnCraftingChange((CraftingManager)null);
    return (ICraftRequest)ct;
  }
  
  public ICraftRequest craftingRequest(ItemStack what) {
    return craftingRequest(what, false, true);
  }
  
  public void advanceCraftingCursor() {
    if (!isPowered())
      return; 
    if (this.waitingThrottle.process()) {
      boolean worked = false;
      Deque p = new LinkedList();
      p.addAll(this.WaitingQueue);
      Iterator<CraftRequest> ix = p.iterator();
      while (ix.hasNext()) {
        CraftRequest x = ix.next();
        if (x.canRequestPrereqs()) {
          ItemStack is = x.getRequest();
          if (logisticsRequest(is)) {
            FMLLog.severe("Just reqyested " + Platform.getItemDisplayName(is) + " - " + is.stackSize, new Object[0]);
            worked = true;
            x.disablePreReqs();
          } 
        } 
      } 
      if (worked)
        this.waitingThrottle.hasAccomplishedWork(); 
    } 
    if (!this.craftingThrottle.process())
      return; 
    List<AssemblerCluster> Clusters = new ArrayList();
    try {
      for (TileRef ar : this.Assemblers) {
        TileAssembler a = (TileAssembler)ar.getTile();
        if (a.ac != null)
          if (Clusters.indexOf(a.ac) == -1)
            Clusters.add(a.ac);  
      } 
      for (AssemblerCluster ac : Clusters) {
        if (useMEEnergy((3 * ac.howManyCpus()), "crafting cpus"))
          ac.cycleCpus(); 
      } 
    } catch (Exception err) {}
    if (this.CraftingQueue.size() == 0)
      return; 
    IMEInventoryHandler iMEInventoryHandler = getCellArray();
    if (iMEInventoryHandler == null)
      return; 
    ItemList all = (ItemList)iMEInventoryHandler.getAvailableItems((IItemList)this.storedList);
    all.clean();
    HashSet Patterns = getPatterns();
    Iterator<CraftRequest> cri = this.WaitingQueue.iterator();
    while (cri.hasNext()) {
      CraftRequest x = cri.next();
      if (x.canTry()) {
        x.clearMissing();
        if (x.getAmount() <= 0)
          continue; 
        ItemStack xRequest = x.getRequest();
        if (xRequest != null) {
          IAssemblerPattern pattern = Crafting.findRecipe(Patterns, xRequest);
          if (x instanceof DelayedCraftRequest) {
            if (pattern != null)
              while (x.getAmount() > 0)
                x.markCrafted();  
            if (x.getAmount() <= 0)
              cri.remove(); 
          } 
        } 
      } 
    } 
    for (int z = 0; z < this.CraftingQueue.size(); z++) {
      CraftRequest x = this.CraftingQueue.pollFirst();
      if (x.canTry()) {
        x.clearMissing();
        if (x.getAmount() > 0) {
          ItemStack xRequest = x.getRequest();
          if (xRequest != null) {
            IAssemblerPattern pattern = Crafting.findRecipe(Patterns, xRequest);
            boolean Recursive = false;
            if (!(x instanceof appeng.me.crafting.MultiPushCraftRequest)) {
              CraftRequest cr = x.getParent();
              while (cr != null) {
                if (!(cr instanceof PushCraftRequest))
                  if (Platform.isSameItem(cr.getRequest(), x.getRequest()))
                    Recursive = true;  
                cr = cr.getParent();
              } 
            } 
            if (Recursive) {
              AppEng.craftingLog(getName(), xRequest, " is recursive");
              this.craftingThrottle.hasAccomplishedWork();
              x.markChainCrafted();
            } else if (pattern == null && x.requirePattern()) {
              if (x.canRequestPrereqs()) {
                AppEng.craftingLog(getName(), xRequest, " is missing, will wait.");
                this.craftingThrottle.hasAccomplishedWork();
                MissingMaterialsCraftRequest missingMaterialsCraftRequest = new MissingMaterialsCraftRequest(getName(), xRequest);
                x.requestedPreReqs((ICraftRequest)missingMaterialsCraftRequest);
                this.WaitingQueue.add(missingMaterialsCraftRequest);
                OnCraftingChange((CraftingManager)null);
              } 
            } else {
              int max = 64;
              while (x.getAmount() > 0) {
                if (max-- < 0)
                  break; 
                IAssemblerCluster ac = null;
                if (pattern != null)
                  ac = pattern.getCluster(); 
                if (ac == null || ac.canCraft())
                  if (x.Craft(this, pattern, (IMEInventory)iMEInventoryHandler, (IItemList)all, this.CraftingQueue, this.WaitingQueue)) {
                    if (ac != null)
                      ac.addCraft(); 
                    this.craftingThrottle.hasAccomplishedWork();
                  }  
              } 
            } 
          } 
          if (x.getAmount() > 0) {
            this.CraftingQueue.addLast(x);
          } else {
            OnCraftingChange((CraftingManager)null);
          } 
        } 
      } else {
        this.CraftingQueue.addLast(x);
      } 
    } 
  }
  
  public WirelessRangeResult inWirelessRange(EntityPlayer p) {
    if (!isPowered())
      return new WirelessRangeResult(null, -1.0F); 
    for (TileRef twr : this.WirelessNodes) {
      try {
        TileWireless tw = (TileWireless)twr.getTile();
        if (((TileEntity)tw).worldObj != ((Entity)p).worldObj)
          continue; 
        double dist = p.getDistanceSq(((TileEntity)tw).xCoord, ((TileEntity)tw).yCoord, ((TileEntity)tw).zCoord);
        double maxRange = AppEngConfiguration.WirelessRange;
        ItemStack boosters = tw.item.getStackInSlot(0);
        if (Platform.isSameItemType(boosters, Materials.matWirelessBooster))
          if (boosters.stackSize > AppEngConfiguration.WirelessRangeExtenders) {
            maxRange += (AppEngConfiguration.WirelessRangeExtenders * AppEngConfiguration.WirelessRangeExtenderBonus);
          } else {
            maxRange += (boosters.stackSize * AppEngConfiguration.WirelessRangeExtenderBonus);
          }  
        if (dist < maxRange * maxRange)
          return new WirelessRangeResult((TileEntity)tw, (float)dist); 
      } catch (AppEngTileMissingException e) {}
    } 
    return new WirelessRangeResult(null, -1.0F);
  }
  
  public void requestUpdate(IGridTileEntity tt) {
    if (!isPowered())
      return; 
    IMEInventoryHandler iMEInventoryHandler = getCellArray();
    if (tt instanceof TileInterfaceBase && iMEInventoryHandler != null) {
      TileInterfaceBase i = (TileInterfaceBase)tt;
      i.loopsSinceUpdate = 0;
      i.reqUpdate = false;
      boolean changed = false;
      for (int l = 0; l < 2; l++) {
        for (int x = 0; x < i.Exports.getSizeInventory(); x++) {
          ItemStack r = i.Exports.getStackInSlot(x);
          ItemStack m = i.getStackInSlot(x);
          if (r == null) {
            if (m != null)
              if (useMEEnergy(m.stackSize, "interface update")) {
                i.setInventorySlotContents(x, Platform.refundEnergy(this, Platform.addItems((IMEInventory)iMEInventoryHandler, m), "interface update"));
                changed = true;
              }  
            continue;
          } 
          if (m != null && !Platform.isSameItem(r, m))
            if (useMEEnergy(m.stackSize, "interface update")) {
              ItemStack result = Platform.refundEnergy(this, Platform.addItems((IMEInventory)iMEInventoryHandler, m), "interface update");
              i.setInventorySlotContents(x, m = result);
              changed = true;
              if (result != null)
                continue; 
            }  
          if (r.stackSize > r.getMaxStackSize())
            r.stackSize = r.getMaxStackSize(); 
          if (m == null) {
            if (useMEEnergy(r.stackSize, "interface update")) {
              ItemStack pre = r.copy();
              i.setInventorySlotContents(x, m = Platform.extractItems((IMEInventory)iMEInventoryHandler, r));
              int diff = Platform.calculateChange(m, pre);
              refundMEEnergy(diff, "interface update");
              if (m != null)
                changed = true; 
            } 
          } else {
            ItemStack diff = r.copy();
            diff.stackSize -= m.stackSize;
            if (diff.stackSize == 0)
              continue; 
            if (diff.stackSize > 0) {
              if (useMEEnergy(diff.stackSize, "interface update")) {
                ItemStack ex = Platform.extractItems((IMEInventory)iMEInventoryHandler, diff);
                int dx = Platform.calculateChange(ex, diff);
                refundMEEnergy(dx, "interface update");
                if (ex != null) {
                  m.stackSize += ex.stackSize;
                  changed = true;
                } 
              } 
            } else {
              diff.stackSize = -diff.stackSize;
              m.stackSize -= diff.stackSize;
              if (useMEEnergy(diff.stackSize, "interface update")) {
                ItemStack un_addable = Platform.refundEnergy(this, Platform.addItems((IMEInventory)iMEInventoryHandler, diff), "Interface update");
                changed = true;
                if (un_addable != null)
                  m.stackSize += un_addable.stackSize; 
              } 
            } 
          } 
          m = i.getStackInSlot(x);
          if (r != null && (m == null || (Platform.isSameItem(r, m) && m.stackSize < r.stackSize))) {
            IItemList avail = getCraftableArray().getAvailableItems();
            IAEItemStack stack = avail.findItem((IAEItemStack)AEItemStack.create(r));
            if (i.craftingMode == InterfaceCraftingMode.Craft && stack != null && stack.isCraftable()) {
              ItemStack req = r.copy();
              req.stackSize = 1;
              i.requestCrafting(req);
            } 
          } 
          continue;
        } 
      } 
      if (changed) {
        triggerContainerUpdate();
        i.onInventoryChanged();
      } 
    } 
  }
  
  public boolean notCrafting(ItemStack i) {
    for (CraftRequest cr : this.CraftingQueue) {
      if (Platform.isSameItemType(cr.getRequest(), i))
        return false; 
    } 
    return true;
  }
  
  public void encodeWireless(ItemStack i) {
    if (i == null)
      return; 
    IWirelessTermHandler handler = AppEng.getApiInstance().getWirelessRegistry().getWirelessTerminalHandler(i);
    if (handler != null)
      handler.setEncryptionKey(i, this.EncryptionKey, getName()); 
  }
  
  boolean inList(HashSet patterns, AssemblerPatternInventory n, IPushable pushable) {
    if (pushable == null)
      return false; 
    for (IAssemblerPattern o : patterns) {
      if (o.equals(n)) {
        o.setInterface(pushable);
        return true;
      } 
    } 
    return false;
  }
  
  HashSet cachedPatternSet = new HashSet();
  
  public HashSet getPatterns() {
    if (this.cachedPatternSet != null)
      return this.cachedPatternSet; 
    HashSet<AssemblerPatternInventory> invs = new HashSet();
    try {
      for (TileRef ar : this.Assemblers) {
        TileAssembler a = (TileAssembler)ar.getTile();
        if (a.ac != null)
          for (int x = 0; x < a.getSizeInventory(); x++) {
            ItemStack s = a.getStackInSlot(x);
            if (s != null && Util.isAssemblerPattern(s).booleanValue()) {
              AssemblerPatternInventory api = (AssemblerPatternInventory)Util.getAssemblerPattern(s);
              if (!inList(invs, api, (IPushable)null)) {
                api.ac = (IAssemblerCluster)a.ac;
                invs.add(api);
              } 
            } 
          }  
      } 
      for (TileRef ar : this.Interfaces) {
        TileInterfaceBase a = (TileInterfaceBase)ar.getTile();
        for (int x = 0; x < a.Crafting.getSizeInventory(); x++) {
          ItemStack s = a.Crafting.getStackInSlot(x);
          if (s != null && Util.isAssemblerPattern(s).booleanValue()) {
            AssemblerPatternInventory api = (AssemblerPatternInventory)Util.getAssemblerPattern(s);
            if (!inList(invs, api, a)) {
              api.setInterface(a);
              invs.add(api);
            } 
          } 
        } 
      } 
    } catch (AppEngTileMissingException err) {
      MinecraftForge.EVENT_BUS.post((Event)new GridErrorEvent(((TileEntity)this).worldObj, getLocation()));
    } 
    return this.cachedPatternSet = invs;
  }
  
  public IMEInventoryHandler getCraftableArray() {
    List<MEInventoryHandler> invs = new ArrayList();
    try {
      for (TileRef ar : this.Interfaces) {
        TileInterfaceBase a = (TileInterfaceBase)ar.getTile();
        for (IMEInventory lp : a.getLogisticsInv())
          invs.add(new MEInventoryHandler(lp)); 
      } 
      CraftingInventory craftingInventory = new CraftingInventory(this, getPatterns());
      invs.add(new MEInventoryHandler((IMEInventory)craftingInventory));
    } catch (AppEngTileMissingException err) {
      MinecraftForge.EVENT_BUS.post((Event)new GridErrorEvent(((TileEntity)this).worldObj, getLocation()));
    } 
    IMEInventoryHandler h = MEInventoryNetwork.getMEInventoryNetwork(invs, this, this.craftList);
    h.setGrid(this);
    return h;
  }
  
  public IMEInventoryHandler getFullCellArray() {
    List<MEInventoryHandler> invs = getCellList();
    try {
      for (TileRef ar : this.Interfaces) {
        TileInterfaceBase a = (TileInterfaceBase)ar.getTile();
        for (IMEInventory lp : a.getLogisticsInv())
          invs.add(new MEInventoryHandler(lp)); 
      } 
      CraftingInventory craftingInventory = new CraftingInventory(this, getPatterns());
      invs.add(new MEInventoryHandler((IMEInventory)craftingInventory));
    } catch (AppEngTileMissingException err) {
      MinecraftForge.EVENT_BUS.post((Event)new GridErrorEvent(((TileEntity)this).worldObj, getLocation()));
    } 
    IMEInventoryHandler h = MEInventoryNetwork.getMEInventoryNetwork(invs, this, this.fullList);
    h.setGrid(this);
    return h;
  }
  
  boolean isCalculatingCell = false;
  
  public List getCellList() {
    this.isCalculatingCell = true;
    List<IMEInventoryHandler> invs = new ArrayList();
    try {
      for (TileRef dr : this.CellContainers) {
        ICellContainer d = (ICellContainer)dr.getTile();
        List lD = d.getCellArray();
        if (lD != null)
          for (IMEInventoryHandler inv : lD) {
            if (inv != null) {
              inv.setPriority(d.getPriority());
              invs.add(inv);
            } 
          }  
      } 
    } catch (AppEngTileMissingException err) {
      MinecraftForge.EVENT_BUS.post((Event)new GridErrorEvent(((TileEntity)this).worldObj, getLocation()));
    } 
    this.isCalculatingCell = false;
    return invs;
  }
  
  IMEInventoryHandler cachedCellArray = null;
  
  static IGridInterface topLevel = null;
  
  public IMEInventoryHandler getCellArray() {
    if (isPowered()) {
      if (this.isCalculatingCell)
        return null; 
      if (this.cachedCellArray != null)
        return this.cachedCellArray; 
      if (topLevel == null)
        topLevel = this; 
      List invs = getCellList();
      IMEInventoryHandler h = MEInventoryNetwork.getMEInventoryNetwork(invs, this, this.storedList);
      h.setGrid(this);
      if (topLevel == this)
        this.cachedCellArray = h; 
      return h;
    } 
    return (IMEInventoryHandler)new MEInventoryNull();
  }
  
  int TicksSincePower = 999;
  
  int TicksSinceUpdate = 0;
  
  int ticksBetweenUpdates = 0;
  
  float cachedPowerConsumption = 0.0F;
  
  boolean isUseingRemotePower = false;
  
  boolean updateStorageAware = true;
  
  private List NewItemQueue;
  
  boolean updatePower() {
    if (this.TicksSincePower++ > 90) {
      this.TicksSincePower = 0;
      return true;
    } 
    return false;
  }
  
  public synchronized void updateTileEntity() {
    if (Platform.isClient())
      return; 
    if (this.TicksBetweenFlashes >= 0)
      this.TicksBetweenFlashes--; 
    if (!this.enabled)
      return; 
    super.updateTileEntity();
    pushPowerUsage();
    try {
      if (updatePower()) {
        this.cachedPowerConsumption = 6.0F + this.cables / 16.0F;
        for (TileRef im : this.AllMachines)
          this.cachedPowerConsumption += ((IGridMachine)im.getTile()).getPowerDrainPerTick(); 
        triggerContainerUpdate();
      } 
      boolean hadPower = ((TileME)this).hasPower;
      boolean hasLocalPower = (this.storedPower > 0.1D);
      ((TileME)this).hasPower = useMEEnergy(this.cachedPowerConsumption, "network drain");
      boolean wasUseingRemotePower = this.isUseingRemotePower;
      this.isUseingRemotePower = (((TileME)this).hasPower && !hasLocalPower);
      if (this.isUseingRemotePower != wasUseingRemotePower)
        this.triggerUpdate = true; 
      if (hadPower != ((TileME)this).hasPower) {
        this.triggerUpdate = true;
        this.sendUpdate = true;
        AppEng.log("Controller " + (((TileME)this).hasPower ? "ONLINE" : "OFFLINE") + "!");
        if (!((TileME)this).hasPower)
          this.storedPower = 0.0F; 
        for (TileRef im : this.AllMachines)
          ((IGridMachine)im.getTile()).setPowerStatus(((TileME)this).hasPower); 
      } 
      if (getPowerLevel() != this.oldFace) {
        this.oldFace = getPowerLevel();
        this.triggerUpdate = true;
      } 
      if (this.triggerUpdate && this.TicksSinceUpdate++ > 5) {
        this.triggerUpdate = false;
        this.TicksSinceUpdate = 0;
        markForUpdate();
      } 
      if (((TileME)this).hasPower) {
        if (this.updateStorageAware) {
          this.updateStorageAware = false;
          updateStorageAware();
        } 
        if (this.NewItemQueue.size() > 0) {
          List ll = this.NewItemQueue;
          this.NewItemQueue = new LinkedList();
          this.sendUpdate = true;
          for (IAEItemStack is : ll) {
            if (is.getStackSize() == 0L)
              continue; 
            long ItemsAdded = is.getStackSize();
            if (is.getStackSize() > 0L) {
              Deque p = new LinkedList();
              p.addAll(this.WaitingQueue);
              Iterator<CraftRequest> ix = p.iterator();
              while (ix.hasNext()) {
                CraftRequest x = ix.next();
                if (x instanceof ExternalCraftRequest)
                  if (is.equals(x.getAERequest())) {
                    while (ItemsAdded > 0L && x.getAmount() > 0) {
                      if (AppEngConfiguration.allowLogging && AppEngConfiguration.logCrafting)
                        AppEng.log(getName() + ": " + x.requestType() + " got " + '\001' + " of " + Platform.getSharedItemStack(is).getItemName() + " - left: " + (x.getAmount() - 1)); 
                      x.markCrafted();
                      ItemsAdded--;
                      OnCraftingChange((CraftingManager)null);
                    } 
                    if (x.getAmount() == 0)
                      this.WaitingQueue.remove(x); 
                    if (ItemsAdded <= 0L)
                      break; 
                  }  
              } 
            } 
          } 
        } 
        for (IGridCache gc : this.Caches)
          gc.onUpdateTick(this); 
        advanceCraftingCursor();
        if (this.sendUpdate || this.hasFlashed)
          if (AppEngConfiguration.gfxCableAnimation)
            if (this.TicksBetweenFlashes < 0) {
              this.hasFlashed = false;
              this.TicksBetweenFlashes = AppEngConfiguration.gfxCableMinTickRate;
              try {
                PacketDispatcher.sendPacketToAllPlayers((Packet)(new PacketGridAnimate(getGridIndex())).getPacket());
              } catch (IOException e) {}
            }   
      } 
      if (this.sendUpdate && this.ticksBetweenUpdates++ > AppEngConfiguration.terminalUpdateMinTickRate) {
        this.sendUpdate = false;
        this.ticksBetweenUpdates = 0;
        Iterator<EntityPlayer> ip = this.contentsViewingPlayers.iterator();
        while (ip.hasNext()) {
          EntityPlayer p = ip.next();
          if (p.openContainer != null) {
            if (p.openContainer instanceof ContainerTerminal) {
              try {
                ContainerTerminal ct = (ContainerTerminal)p.openContainer;
                ct.GetNetworkIME().postChanges(this.invChanges);
                for (Packet250CustomPayload pak : ct.GetNetworkIME().getDataPacket())
                  PacketDispatcher.sendPacketToPlayer((Packet)pak, (Player)p); 
              } catch (IOException e) {
                e.printStackTrace();
              } 
              continue;
            } 
            ip.remove();
          } 
        } 
        this.invChanges = new ArrayList();
      } 
    } catch (AppEngTileMissingException e) {
    
    } catch (ClassCastException e) {
      MinecraftForge.EVENT_BUS.post((Event)new GridErrorEvent(((TileEntity)this).worldObj, getLocation()));
    } 
  }
  
  private void updateStorageAware() throws AppEngTileMissingException {
    ItemList iss = (ItemList)getCellArray().getAvailableItems((IItemList)this.storedList);
    for (TileRef sm : this.StorageAware) {
      IStorageAware tsm = (IStorageAware)sm.getTile();
      tsm.onNetworkInventoryChange((IItemList)iss);
    } 
  }
  
  public void addViewingPlayer(EntityPlayer p) {
    if (this.contentsViewingPlayers.indexOf(p) == -1)
      this.contentsViewingPlayers.add(p); 
  }
  
  public void rmvViewingPlayer(EntityPlayer p) {
    this.contentsViewingPlayers.remove(p);
  }
  
  private static int newGridIndex = 1;
  
  boolean powerBar;
  
  IGridInterface myRef;
  
  public String getName() {
    return this.EncryptionKey;
  }
  
  public int getPowerLevel() {
    int powerLevel = (int)Math.ceil((5.0F * this.storedPower / this.maxStoredPower));
    if (powerLevel < 0)
      powerLevel = 0; 
    if (powerLevel >= AppEngTextureRegistry.Blocks.MEControllerPower.length)
      powerLevel = AppEngTextureRegistry.Blocks.MEControllerPower.length - 1; 
    return powerLevel;
  }
  
  public TileController() {
    this.powerBar = false;
    this.myRef = null;
    this.gridIndex = newGridIndex++;
    this.Caches = AppEng.getApiInstance().getGridCacheRegistry().createCacheInstance();
    this.fullList = new ItemList();
    this.craftList = new ItemList();
    this.storedList = new ItemList();
    this.cables = 0;
    this.maxStoredPower = 4000.0F;
    this.currentStatus = new ArrayList();
    this.invChanges = new ArrayList();
    this.CellContainers = new ArrayList();
    this.Assemblers = new ArrayList();
    this.Interfaces = new ArrayList();
    this.WirelessNodes = new ArrayList();
    this.StorageAware = new ArrayList();
    this.AllMachines = new ArrayList();
    this.PowerSources = new ArrayList();
    this.WaitingQueue = new LinkedList();
    this.CraftingQueue = new LinkedList();
    this.contentsViewingPlayers = new ArrayList();
    this.craftingViewingPlayers = new ArrayList();
    this.personalPowerUsageLog = new float[20];
    this.realPowerUsageLog = new float[20];
    this.updateStorageAware = true;
    this.NewItemQueue = new LinkedList();
    this.CManager = new CraftingManager(getName(), this);
    this.EncryptionKey = String.valueOf((new Date()).getTime());
    this.orientation = ForgeDirection.NORTH;
    this.waitingThrottle = new METhrottle(40, AppEngConfiguration.craftingMinTickRate, AppEngConfiguration.craftingMinTickRate + 320);
    this.craftingThrottle = new METhrottle(1, AppEngConfiguration.craftingMinTickRate, AppEngConfiguration.craftingMinTickRate + 80);
  }
  
  public Icon getFrontFace() {
    if (!AppEngConfiguration.requirePower)
      return AppEngTextureRegistry.Blocks.GenericSide.get(); 
    if (this.powerBar)
      return getPowerLevelBar(); 
    return AppEngTextureRegistry.Blocks.ControllerPanel.get();
  }
  
  public Icon getPowerLevelBar() {
    if (!AppEngConfiguration.requirePower)
      return null; 
    if (this.isUseingRemotePower && ((TileME)this).hasPower)
      return AppEngTextureRegistry.Blocks.BlockControllerLinked.get(); 
    return AppEngTextureRegistry.Blocks.MEControllerPower[getPowerLevel()].get();
  }
  
  public Icon getBlockTextureFromSide(ForgeDirection side) {
    if (ForgeDirection.DOWN == side)
      return AppEngTextureRegistry.Blocks.GenericBottom.get(); 
    if (ForgeDirection.UP == side)
      return AppEngTextureRegistry.Blocks.GenericTop.get(); 
    Icon frontFace = getFrontFace();
    if (this.orientation == side)
      return frontFace; 
    return AppEngTextureRegistry.Blocks.GenericSide.get();
  }
  
  public IGridInterface getGrid() {
    return this;
  }
  
  public void craftGui(EntityPlayerMP pmp, IGridTileEntity gte, ItemStack s) {
    HashSet patterns = getPatterns();
    IAssemblerPattern p = Crafting.findRecipe(patterns, s);
    if (p != null || canLogisticsMake(s)) {
      this.lastCraftingRequest = s;
      WorldCoord wc = gte.getLocation();
      Platform.openGui((EntityPlayer)pmp, AppEngGuiHandler.GUI_CRAFTING, ((Entity)pmp).worldObj, wc.x, wc.y, wc.z);
    } 
  }
  
  public void configureController(Collection nodes) {
    if (Platform.isClient())
      return; 
    this.CManager = new CraftingManager(getName(), this);
    this.CraftingQueue.clear();
    this.WaitingQueue.clear();
    this.cables = 0;
    this.updateStorageAware = true;
    this.sendUpdate = true;
    this.currentStatus = null;
    this.cachedPatternSet = null;
    this.cachedCellArray = null;
    this.CellContainers.clear();
    this.Assemblers.clear();
    this.Interfaces.clear();
    this.WirelessNodes.clear();
    this.StorageAware.clear();
    this.AllMachines.clear();
    this.AllEntities = new ArrayList();
    this.PowerSources = new ArrayList();
    triggerPowerUpdate();
    if (nodes == null) {
      this.enabled = false;
      this.currentStatus = null;
      this.cables = 0;
      this.triggerUpdate = true;
      return;
    } 
    List<IGridTileEntity> currentTiles = new ArrayList();
    if (this.myRef == null)
      this.myRef = (IGridInterface)new GridReference(this); 
    for (GridEnumeration.NetworkNode tt : nodes) {
      IGridTileEntity gte = tt.getTile();
      currentTiles.add(gte);
      this.AllEntities.add(new TileRef((TileEntity)gte));
      gte.setGrid(this.myRef);
      gte.setPowerStatus(((TileME)this).hasPower);
    } 
    this.enabled = true;
    try {
      this.currentStatus = new ArrayList();
      for (TileRef tr : this.AllEntities) {
        TileEntity tt = (TileEntity)tr.getTile();
        if (tt instanceof TileCable || tt instanceof TileColorlessCable) {
          Platform.sumItemToList(this.currentStatus, Blocks.blkColorlessCable);
        } else {
          Platform.sumItemToList(this.currentStatus, Platform.getItemStackVersion(tt.worldObj, tt.xCoord, tt.yCoord, tt.zCoord));
        } 
        if (tt instanceof TileCable || tt instanceof TileDarkCable || tt instanceof TileColorlessCable)
          this.cables++; 
        if (tt instanceof IGridMachine)
          this.AllMachines.add(new TileRef(tt)); 
        if (tt instanceof IStorageAware)
          this.StorageAware.add(new TileRef(tt)); 
        if (tt instanceof ICellContainer)
          this.CellContainers.add(new TileRef(tt)); 
        if (tt instanceof TileWireless)
          this.WirelessNodes.add(new TileRef(tt)); 
        if (tt instanceof TileAssembler)
          this.Assemblers.add(new TileRef(tt)); 
        if (tt instanceof TileInterfaceBase)
          this.Interfaces.add(new TileRef(tt)); 
        if (tt instanceof IPowerSharing)
          this.PowerSources.add(new TileRef(tt)); 
      } 
      for (TileRef tt : this.Interfaces)
        requestUpdate((IGridTileEntity)tt.getTile()); 
      AppEng.log("Online: " + this.AllMachines.size());
      updateStorageAware();
      getCraftableArray().getAvailableItems();
      getFullCellArray().getAvailableItems();
      this.triggerUpdate = true;
    } catch (AppEngTileMissingException e) {
      configureController((Collection)null);
      return;
    } 
    for (IGridCache gc : this.Caches)
      gc.reset(this); 
  }
  
  public void writeToNBT(NBTTagCompound data) {
    super.writeToNBT(data);
    data.setInteger("rot", getAERotationFromDirection(this.orientation));
    data.setString("encKey", this.EncryptionKey);
    for (IGridCache gc : this.Caches) {
      NBTTagCompound tc = gc.savetoNBTData();
      if (tc != null)
        data.setCompoundTag("GC." + gc.getCacheName(), tc); 
    } 
  }
  
  public void readFromNBT(NBTTagCompound data) {
    super.readFromNBT(data);
    this.orientation = getDirectionFromAERotation((byte)data.getInteger("rot"));
    this.EncryptionKey = data.getString("encKey");
    if (this.EncryptionKey == null || this.EncryptionKey == "")
      this.EncryptionKey = String.valueOf((new Date()).getTime()); 
    for (IGridCache gc : this.Caches) {
      NBTTagCompound o = data.getCompoundTag("GC." + gc.getCacheName());
      gc.loadfromNBTData(o);
    } 
  }
  
  public void configureTilePacket(DataOutputStream data) {
    try {
      data.writeBoolean(((TileME)this).hasPower);
      data.writeBoolean(this.enabled);
      data.writeFloat(this.storedPower);
      data.writeBoolean(this.isUseingRemotePower);
      data.writeByte(getAERotationFromDirection(this.orientation));
      data.writeFloat(getPowerUsageAvg());
      data.writeFloat(getPersonalPowerUsageAvg());
      if (this.currentStatus == null) {
        data.writeShort(0);
      } else {
        data.writeShort((short)this.currentStatus.size());
        for (int x = 0; x < this.currentStatus.size(); x++) {
          data.writeInt(((ItemStack)this.currentStatus.get(x)).itemID);
          data.writeInt(((ItemStack)this.currentStatus.get(x)).stackSize);
          data.writeInt(((ItemStack)this.currentStatus.get(x)).getItemDamage());
        } 
      } 
    } catch (IOException e) {
      return;
    } 
  }
  
  public boolean isValid() {
    return true;
  }
  
  public void handleTilePacket(DataInputStream stream) {
    try {
      ForgeDirection oldOrientation = this.orientation;
      float oldstoredPower = this.storedPower;
      boolean wasUseingRemotePower = this.isUseingRemotePower;
      ((TileME)this).hasPower = stream.readBoolean();
      this.enabled = stream.readBoolean();
      this.storedPower = stream.readFloat();
      this.isUseingRemotePower = stream.readBoolean();
      this.orientation = getDirectionFromAERotation(stream.readByte());
      this.powerDrained = stream.readFloat();
      this.powerDrainedPersonal = stream.readFloat();
      short statusCount = stream.readShort();
      if (statusCount > 0) {
        this.currentStatus = new ArrayList();
        for (int x = 0; x < statusCount; x++)
          this.currentStatus.add(new ItemStack(stream.readInt(), stream.readInt(), stream.readInt())); 
        Collections.sort(this.currentStatus, ItemSorters.Decending_SortBySize_Vanilla);
      } else {
        this.currentStatus = null;
      } 
      if (Math.abs(this.storedPower - oldstoredPower) > 0.05D || oldOrientation != this.orientation || this.isUseingRemotePower != wasUseingRemotePower)
        markForUpdate(); 
    } catch (IOException e) {
      return;
    } 
  }
  
  public IMEInventoryHandler provideCell() {
    if (((TileME)this).hasPower)
      return getFullCellArray(); 
    return null;
  }
  
  public boolean useMEEnergy(float use, String for_what) {
    if (!AppEngConfiguration.requirePower)
      return true; 
    if (AppEngConfiguration.allowLogging && AppEngConfiguration.logPowerUsage)
      AppEng.log(getName() + ": " + use + " units for " + for_what); 
    this.personalPowerUsageActiveTick += use;
    return useMEEnergyRecursive(use, new ArrayList());
  }
  
  List getPowerSources() {
    List<Object> providers = new ArrayList();
    for (TileRef i : this.Interfaces) {
      try {
        TileInterfaceBase tib = (TileInterfaceBase)i.getTile();
        for (Object lpp : tib.getLogisticsPowerSources())
          providers.add(lpp); 
      } catch (AppEngTileMissingException e) {
        AppEng.log("Error accessing Interface.");
      } 
    } 
    for (TileRef irpp : this.PowerSources) {
      try {
        providers.add(irpp.getTile());
      } catch (AppEngTileMissingException e) {
        AppEng.log("Error accessing PowerSource.");
      } 
    } 
    return providers;
  }
  
  public boolean useMEEnergyRecursive(float amount, List<TileController> providersToIgnore) {
    if (!providersToIgnore.contains(this)) {
      providersToIgnore.add(this);
      if (this.storedPower >= amount)
        return useLocalMEEnergy(amount); 
      this.storedPower = 0.0F;
      for (Object pp : getPowerSources()) {
        if (pp instanceof IPowerSharing) {
          if (((IPowerSharing)pp).useEnergy((int)Math.ceil(amount), providersToIgnore))
            return true; 
          continue;
        } 
        if ((AppEng.getInstance()).LPProxy != null)
          if ((AppEng.getInstance()).LPProxy.canUseEnergy(pp, (int)Math.ceil(amount), providersToIgnore) && 
            (AppEng.getInstance()).LPProxy.useEnergy(pp, (int)Math.ceil(amount), providersToIgnore))
            return true;  
      } 
    } 
    return false;
  }
  
  public boolean canUseEnergyRecursive(float amount, List<TileController> providersToIgnore) {
    if (!providersToIgnore.contains(this)) {
      if (this.storedPower >= amount)
        return true; 
      providersToIgnore.add(this);
      for (Object pp : getPowerSources()) {
        if (pp instanceof IPowerSharing) {
          if (((IPowerSharing)pp).canUseEnergy((int)Math.ceil(amount), providersToIgnore))
            return true; 
          continue;
        } 
        if ((AppEng.getInstance()).LPProxy != null)
          if ((AppEng.getInstance()).LPProxy.canUseEnergy(pp, (int)Math.ceil(amount), providersToIgnore))
            return true;  
      } 
    } 
    return false;
  }
  
  public void refundMEEnergy(float use, String for_what) {
    if (!AppEngConfiguration.requirePower)
      return; 
    if (use < 0.0F)
      return; 
    use *= AppEngConfiguration.powerUsageMultiplier;
    this.storedPower += use;
    this.realPowerUsageActiveTick -= use;
    this.personalPowerUsageActiveTick -= use;
  }
  
  public boolean useLocalMEEnergy(float use) {
    if (!AppEngConfiguration.requirePower)
      return true; 
    use *= AppEngConfiguration.powerUsageMultiplier;
    this.realPowerUsageActiveTick += use;
    if (this.storedPower >= use) {
      this.storedPower -= use;
      return true;
    } 
    this.storedPower = 0.0F;
    return false;
  }
  
  public TileEntity getController() {
    return (TileEntity)this;
  }
  
  public void placedBy(EntityLiving entityliving) {
    this.orientation = getOrientationFromLivingEntity(entityliving, false);
  }
  
  public int usePowerForAddition(int items, int multiplier) {
    if (!AppEngConfiguration.requirePower)
      return items; 
    if (!((TileME)this).hasPower)
      return 0; 
    if (this.storedPower > (items * multiplier))
      return useMEEnergy(items, "adding items") ? items : 0; 
    if (useMEEnergy((items * multiplier), "adding items"))
      return items; 
    int amt = (int)Math.floor((this.storedPower * multiplier));
    return useMEEnergy(amt, "adding items") ? (amt / 2) : 0;
  }
  
  protected int getMaxStoredPower() {
    return (int)this.maxStoredPower;
  }
  
  public ItemStack getCraftAmount(ItemStack willAdd) {
    IAssemblerPattern p = Crafting.findRecipe(getPatterns(), willAdd);
    if (p != null)
      return p.getOutput(); 
    return null;
  }
  
  public List getParts() {
    if (this.currentStatus == null)
      return new ArrayList(); 
    return this.currentStatus;
  }
  
  public List getMachines() {
    List<TileRef> mac = new ArrayList();
    for (TileRef r : this.AllMachines) {
      try {
        mac.add(new TileRef((TileEntity)r.getTile()));
      } catch (AppEngTileMissingException e) {}
    } 
    return mac;
  }
  
  public void addCraftingPlayer(EntityPlayer p) {
    if (this.craftingViewingPlayers.indexOf(p) == -1)
      this.craftingViewingPlayers.add(p); 
  }
  
  public void rmvCraftingPlayer(EntityPlayer p) {
    this.craftingViewingPlayers.remove(p);
  }
  
  public boolean syncStyle(IAppEngNetworkTile.SyncTime st) {
    return true;
  }
  
  public void removeFromCraftingQueues(CraftRequest craftRequest) {
    this.CraftingQueue.remove(craftRequest);
    this.WaitingQueue.remove(craftRequest);
  }
  
  public IMEInventoryHandler provideCell(String Filter) {
    if (((TileME)this).hasPower) {
      if (Filter.equals(ViewItems.ALL.toString()))
        return getFullCellArray(); 
      if (Filter.equals(ViewItems.CRAFTABLE.toString()))
        return getCraftableArray(); 
      if (Filter.equals(ViewItems.STORED.toString()))
        return getCellArray(); 
    } 
    return null;
  }
  
  public int getGridIndex() {
    return this.gridIndex;
  }
  
  public IAssemblerPattern getPatternFor(ItemStack req) {
    for (IAssemblerPattern p : getPatterns()) {
      if (Platform.isSameItem(req, p.getOutput()))
        return p; 
    } 
    return null;
  }
  
  public void notifyExtractItems(IAEItemStack removed) {
    this.sendUpdate = true;
    IAEItemStack ns = removed.copy();
    ns.setStackSize(-ns.getStackSize());
    Platform.sumItemToList(this.invChanges, ns);
    this.updateStorageAware = true;
  }
  
  public void notifyAddItems(IAEItemStack added) {
    this.sendUpdate = true;
    IAEItemStack ns = added.copy();
    Platform.sumItemToList(this.NewItemQueue, ns);
    Platform.sumItemToList(this.invChanges, ns);
    this.updateStorageAware = true;
    this.craftingThrottle.wakeUp();
  }
  
  public void triggerPowerUpdate() {
    this.TicksSincePower = 900;
  }
  
  public long getLocatableSerial() {
    return Long.valueOf(this.EncryptionKey).longValue();
  }
  
  public IGridCache getCacheByID(int id) {
    return this.Caches[id];
  }
  
  public void signalEnergyTransfer(IGridTileEntity a, IGridTileEntity b, float amt) {
    this.hasFlashed = true;
  }
  
  public ForgeDirection getPrimaryOrientation() {
    return this.orientation;
  }
  
  public int getSpin() {
    return 0;
  }
  
  public void setPrimaryOrientation(ForgeDirection s) {
    this.orientation = s;
  }
  
  public void setSpin(int spin) {}
}
