//date: 2022-01-05T17:05:25Z
//url: https://api.github.com/gists/4c50da55db48778cb936705ae6755de4
//owner: https://api.github.com/users/timebertt

// json merge patch + optimistic locking
patch := client.MergeFromWithOptions(shoot.DeepCopy(), client.MergeFromWithOptimisticLock{})
// ...

// strategic merge patch + optimistic locking
patch = client.StrategicMergeFrom(shoot.DeepCopy(), client.MergeFromWithOptimisticLock{})
// ...