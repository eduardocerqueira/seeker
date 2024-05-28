//date: 2024-05-28T17:07:57Z
//url: https://api.github.com/gists/898c80e3b86853efb22bfb84d8759c27
//owner: https://api.github.com/users/Deep-Coder-zhui

// 任务添加和获取
// 下面代码的作用是将需要执行的任务添加到队列中，并通过线程池中的线程执行这些任务，以实现异步任务处理。

// 存放任务的队列定义
// 队列采用了阻塞式的队列，即当队列已满时，生产者线程会被阻塞直到队列有空间为止。
private final LinkedBlockingQueue<CommandContext> jobExecutionQueue = new LinkedBlockingQueue<>();

// 添加需要运行的任务到队列中
public void addKillCommand(Long jobExecutionId) {
        CommandContext commandContext = new CommandContext();
        commandContext.setCommandCode(CommandCode.JOB_KILL_REQUEST);
        commandContext.setJobExecutionId(jobExecutionId);
	
        jobExecutionQueue.offer(commandContext);
}

// 获取队列里面的任务进行处理
class JobExecutionExecutor implements Runnable {
    @Override
    public void run() {
		// 使用一个循环来持续监听队列中是否有任务需要执行。
        while (Stopper.isRunning()) {
            try {
				// 如果队列为空，take 方法会阻塞当前线程直到队列中有任务为止。一旦有任务可执行，
				// 就会取出任务并进行处理。
                CommandContext commandContext = jobExecutionQueue.take();
				
                Long jobExecutionId = commandContext.getJobExecutionId();
                JobExecutionRequest jobExecutionRequest = commandContext.getJobExecutionRequest();
				
                if (unFinishedJobExecutionMap.get(jobExecutionId) == null) {
                    continue;
                }
				// 在任务处理的过程中，会根据任务的类型执行不同的操作。
				// 如果是 JOB_EXECUTE_REQUEST 类型的任务，会执行 executeJobExecution 方法，
				// 并在一定时间后设置任务的超时处理。如果是 JOB_KILL_REQUEST 类型的任务，
				// 则会执行 doKillCommand 方法。
                switch(commandContext.getCommandCode()){
                    case JOB_EXECUTE_REQUEST:
                        executeJobExecution(jobExecutionRequest);
                        wheelTimer.newTimeout(
                                new JobExecutionTimeoutTimerTask(
                                        jobExecutionId, jobExecutionRequest.getRetryTimes()),
                                jobExecutionRequest.getTimeout()+2, TimeUnit.SECONDS);
                        break;
                    case JOB_KILL_REQUEST:
						// 用于处理任务的终止请求，实际上是从一个 Map 中移除任务的相关信息。
                        doKillCommand(jobExecutionId);
                        break;
                    default:
                        break;
                }

            } catch(Exception e) {
                logger.error("dispatcher job error",e);
				// 在捕获到异常后让当前线程休眠 2 秒钟。这样做的目的可能是为了避免异常频繁发生时对系统造成过大的负担，
				// 通过线程休眠一段时间来减缓异常的发生频率，同时也可以防止异常造成线程资源的浪费。
                ThreadUtils.sleep(2000);
            }
        }
    }
}