#date: 2022-06-24T17:14:52Z
#url: https://api.github.com/gists/1139334c5e68c7db420c5b64df6bcb6e
#owner: https://api.github.com/users/prithvidiamond1

from copy import deepcopy

class User:
    name = ""
    jobPreferenceIndex = 0
    quality = 0

    def __init__(self, name, jobPreferenceIndex, quality):
        self.name = name
        self.jobPreferenceIndex = jobPreferenceIndex
        self.quality = quality


class Job:
    title = ""
    budget = 0

    def __init__(self, title, budget):
        self.title = title
        self.budget = budget


class JobAssignment:
    Users = {}
    Jobs = {}

    def assignJobToUser(self, user, job):
        self.Users[user.name] = job
        if user in self.Jobs[job.title]:
            self.Jobs[job.title].append(user)
        else:
            self.Jobs[job.title] = user


def sja(users, jobs):
    jobAssignment = JobAssignment()
    availableJobs = deepcopy(jobs)
    i = 0
    while (i < len(users)) and (users[i] not in jobAssignment) and (len(availableJobs) != 0):
        preferredJobIndex = users[i].jobPreferenceIndex  # Users most preferred job number
        preferredJob = availableJobs.pop(preferredJobIndex)  # selected preferred job and removed it from available Jobs in a single step
        if preferredJob.budget >= users[i].quality:
            jobAssignment.assignJobToUser(users[i], preferredJob)   # refer to assignJobToUser method for implementation details
            preferredJob.budget -= users[i].quality
        else:
            lowerQualityUsers = [user for user in jobAssignment.Jobs[preferredJob.title] if users[i].quality > user.quality]
            acceptableUsers =  # sets of users from the low quality set that are acceptable
            if len(acceptableUsers) != 0:



        i += 1
    return jobAssignment


testUser = User("Gustavo", 1, 4)
testJob = Job("Engineer", 4)

sampleUsers = [testUser]
sampleJobs = [testJob]

assignment = sja(sampleUsers, sampleJobs)
