//date: 2023-06-05T17:01:31Z
//url: https://api.github.com/gists/a883e532c622edce45aab61b4a3071ab
//owner: https://api.github.com/users/anavarr

static {
  try {
    Lookup lookup = MethodHandles.lookup();
    Lookup privateLookupIn = MethodHandles.privateLookupIn(Thread.class, lookup);
    getCurrentCarrierHandle = privateLookupIn.findStatic(Thread.class, "currentCarrierThread", MethodType.methodType(Thread.class));
    isVirtualHandle = privateLookupIn.findVirtual(Thread.class, "isVirtual", MethodType.methodType(Boolean.TYPE));
  } catch (Throwable e) {
    throw e;
  }
  threadCaches = new ConcurrentHashMap();
}

protected ByteBuf newDirectBuffer(int initialCapacity, int maxCapacity) {
   boolean isVirtual = false;

   try {
      isVirtual = (boolean)isVirtualHandle.invokeExact(Thread.currentThread());
   } catch (InvocationTargetException | NoSuchMethodException | ClassNotFoundException | IllegalAccessException var7) {
      System.out.println("error in newDirectBuffer : ");
      System.err.println(var7);
   }

   PoolThreadCache cache = null;
   if (isVirtual) {
      cache = this.createCache();
   }

   if (cache == null) {
      cache = (PoolThreadCache)this.threadCache.get();
   }

   PoolArena directArena = cache.directArena;
   Object buf;
   if (directArena != null) {
      buf = directArena.allocate(cache, initialCapacity, maxCapacity);
   } else {
      buf = PlatformDependent.hasUnsafe()
         ? PlatformDependent.newUnsafeDirectByteBuf(this, initialCapacity, maxCapacity)
         : new UnpooledDirectByteBuf(this, initialCapacity, maxCapacity);
   }

   return toLeakAwareBuffer((ByteBuf)buf);
}

private PoolThreadCache createCache() {
   Thread currentCarrierThread = Thread.currentThread();

   try {
      currentCarrierThread = (Thread)getCurrentCarrierHandle.invokeExact();
   } catch (ClassNotFoundException | InvocationTargetException | IllegalAccessException | NoSuchMethodException var11) {
      return null;
   }

   PoolThreadCache cache = (PoolThreadCache)null;
   if (((ConcurrentHashMap)threadCaches).containsKey(currentCarrierThread)) {
      return (PoolThreadCache)((ConcurrentHashMap)threadCaches).get(currentCarrierThread);
   } else {
      PoolArena<byte[]> heapArena = this.leastUsedArena(this.heapArenas);
      PoolArena<byte[]> directArena = this.leastUsedArena(this.directArenas);
      cache = new PoolThreadCache(
         heapArena, directArena, this.smallCacheSize, this.normalCacheSize, DEFAULT_MAX_CACHED_BUFFER_CAPACITY, DEFAULT_CACHE_TRIM_INTERVAL
      );
      ((ConcurrentHashMap)threadCaches).put(currentCarrierThread, cache);
      if (DEFAULT_CACHE_TRIM_INTERVAL_MILLIS > 0L) {
         EventExecutor executor = ThreadExecutorMap.currentExecutor();
         if (executor != null) {
            executor.scheduleAtFixedRate(this.trimTask, DEFAULT_CACHE_TRIM_INTERVAL_MILLIS, DEFAULT_CACHE_TRIM_INTERVAL_MILLIS, MILLISECONDS);
         }
      }

      return cache;
   }
}