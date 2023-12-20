#date: 2023-12-20T16:50:22Z
#url: https://api.github.com/gists/63771004de6752b361b0c0361e527324
#owner: https://api.github.com/users/matt555

# time command must be given full path to avoid using bash builtin

# basic example
/usr/bin/time -f "\nTime:\n\t%E\treal\n\t%U\tuser\n\t%S\tsys\n\t%P\t%%cpu\n" -- sleep 1

# single string with all options
timeFormat='\nTime:\n\t%E\tElapsed real time (hours:minutes:seconds).\n\t%e\tElapsed real time (seconds).\n\t%S\tTotal CPU-seconds that the process spent in kernel mode.\n\t%U\tTotal CPU-seconds that the process spent in user mode.\n\t%P\tPercentage of the CPU that this job got.\n\nMemory:\n\t%M\tMaximum resident set size of the process during its lifetime, in Kbytes.\n\t%t\tAverage resident set size of the process, in Kbytes.\n\t%K\tAverage total (data+stack+text) memory use of the process, in Kbytes.\n\t%D\tAverage size of the processs unshared data area, in Kbytes.\n\t%p\tAverage size of the processs unshared stack space, in Kbytes.\n\t%X\tAverage size of the processs shared text space, in Kbytes.\n\t%Z\tSystems page size, in bytes.  This is a per-system constant, but varies between systems.\n\t%F\tNumber of major page faults that occurred while the process was running.  These are faults where the page has to be read in from disk.\n\t%R\tNumber of minor, or recoverable, page faults.  These are faults for pages that are not valid but which have not yet been claimed by other virtual pages.  Thus the data in the page is still valid but the system tables must be updated.\n\t%W\tNumber of times the process was swapped out of main memory.\n\t%c\tNumber of times the process was context-switched involuntarily (because the time slice expired).\n\t%w\tNumber of waits: times that the program was context-switched voluntarily, for instance while waiting for an I/O operation to complete.\n\nI/O:\n\t%I\tNumber of filesystem inputs by the process.\n\t%O\tNumber of filesystem outputs by the process.\n\t%r\tNumber of socket messages received by the process.\n\t%s\tNumber of socket messages sent by the process.\n\t%k\tNumber of signals delivered to the process.\n\t%x\tExit status of the command.\n\nCommand:\n\t%C\n'

/usr/bin/time -f "${timeFormat}" -- sleep 1


# for use in scripts, easier to reorder/remove things:
t_time='\nTime:\n\t%E\tElapsed real time (hours:minutes:seconds).\n\t%e\tElapsed real time (seconds).\n\t%S\tTotal CPU-seconds that the process spent in kernel mode.\n\t%U\tTotal CPU-seconds that the process spent in user mode.\n\t%P\tPercentage of the CPU that this job got.\n'

t_memory='\nMemory:\n\t%M\tMaximum resident set size of the process during its lifetime, in Kbytes.\n\t%t\tAverage resident set size of the process, in Kbytes.\n\t%K\tAverage total (data+stack+text) memory use of the process, in Kbytes.\n\t%D\tAverage size of the processs unshared data area, in Kbytes.\n\t%p\tAverage size of the processs unshared stack space, in Kbytes.\n\t%X\tAverage size of the processs shared text space, in Kbytes.\n\t%Z\tSystems page size, in bytes.  This is a per-system constant, but varies between systems.\n\t%F\tNumber of major page faults that occurred while the process was running.  These are faults where the page has to be read in from disk.\n\t%R\tNumber of minor, or recoverable, page faults.  These are faults for pages that are not valid but which have not yet been claimed by other virtual pages.  Thus the data in the page is still valid but the system tables must be updated.\n\t%W\tNumber of times the process was swapped out of main memory.\n\t%c\tNumber of times the process was context-switched involuntarily (because the time slice expired).\n\t%w\tNumber of waits: times that the program was context-switched voluntarily, for instance while waiting for an I/O operation to complete.\n'

t_io='\nI/O:\n\t%I\tNumber of filesystem inputs by the process.\n\t%O\tNumber of filesystem outputs by the process.\n\t%r\tNumber of socket messages received by the process.\n\t%s\tNumber of socket messages sent by the process.\n\t%k\tNumber of signals delivered to the process.\n\t%x\tExit status of the command.\n'

t_command='\nCommand:\n\t%C\n'

timeFormat="${t_time}${t_memory}${t_io}${t_command}"

/usr/bin/time -f "${timeFormat}" -- sleep 1


# save to file:
/usr/bin/time -f "${timeFormat}" -o time_output.txt -- sleep 1

# no output, save to file:
/usr/bin/time -f "${timeFormat}" -o time_output.txt -- ls -lU --color=none > /dev/null 2>&1

