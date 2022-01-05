//date: 2022-01-05T17:03:34Z
//url: https://api.github.com/gists/3d82c1233b1ef1703ebd9a0748d276fd
//owner: https://api.github.com/users/timebertt

var (
  ctx   context.Context
  c     client.Client
  shoot *gardencorev1beta1.Shoot
)

// update
shoot.Spec.Kubernetes.Version = "1.22"
err := c.Update(ctx, shoot)

// json merge patch
patch := client.MergeFrom(shoot.DeepCopy())
shoot.Spec.Kubernetes.Version = "1.22"
err = c.Patch(ctx, shoot, patch)

// strategic merge patch
patch = client.StrategicMergeFrom(shoot.DeepCopy())
shoot.Spec.Kubernetes.Version = "1.22"
err = c.Patch(ctx, shoot, patch)