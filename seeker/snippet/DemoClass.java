//date: 2022-09-09T16:59:13Z
//url: https://api.github.com/gists/4a8817e8cdf31816813ed47cc47c2f48
//owner: https://api.github.com/users/vishalratna-microsoft

package com.tutorials.vishal;

import com.tutorials.vishal.hierarchy.BaseWork;
import com.tutorials.vishal.hierarchy.RxWork;
import com.tutorials.vishal.hierarchy.Scheduler;

public class DemoClass {

    void driver() {
        BaseWork work = new BaseWork();
        RxWork rxWork = new RxWork();
        
        // We are able to submit all types of work. Yay!
        startJob(work); // Valid!
        startJob(rxWork); // Valid, polymorphism magic.
    }
    void startJob(BaseWork incomingWork) {
        // validate the work and submit. 
        Scheduler.submit(incomingWork);
    }
}
