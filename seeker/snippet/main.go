//date: 2025-05-23T16:40:26Z
//url: https://api.github.com/gists/dec6397e621f3b84983478105502871f
//owner: https://api.github.com/users/jveski

package main

import (
	"context"
	"log/slog"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	flowv1 "k8s.io/api/flowcontrol/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/metrics/server"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

// pid tunings
const (
	kP          = 1    // p
	kI          = 0.05 // i
	kD          = 0.1  // d
	minShares   = 1
	maxShares   = 200
	maxIntegral = 1000.0
)

func main() {
	c := &controller{
		lastSamples: make(map[types.NamespacedName]time.Time),
		lastStates:  make(map[types.NamespacedName]corev1.ConditionStatus),
	}
	startOuterController(c)
	startInnerController(c)
	<-context.Background().Done()
}

func startOuterController(c *controller) {
	mgrOpts := manager.Options{Logger: logr.FromSlogHandler(slog.Default().Handler()), Metrics: server.Options{BindAddress: "0"}}
	ctrl.SetLogger(mgrOpts.Logger)
	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), mgrOpts)
	if err != nil {
		panic(err)
	}

	c.outerClient = mgr.GetClient()
	err = ctrl.NewControllerManagedBy(mgr).For(&corev1.Pod{}).Complete(reconcile.Func(c.ReconcilePod))
	if err != nil {
		panic(err)
	}

	go func() {
		err = mgr.Start(context.Background())
		if err != nil {
			panic(err)
		}
	}()
}

func startInnerController(c *controller) {
	rc, err := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
		&clientcmd.ClientConfigLoadingRules{ExplicitPath: os.Getenv("INNER_KUBECONFIG")},
		&clientcmd.ConfigOverrides{}).ClientConfig()
	if err != nil {
		panic(err)
	}

	mgrOpts := manager.Options{Logger: logr.FromSlogHandler(slog.Default().Handler()), Metrics: server.Options{BindAddress: "0"}}
	ctrl.SetLogger(mgrOpts.Logger)
	mgr, err := ctrl.NewManager(rc, mgrOpts)
	if err != nil {
		panic(err)
	}

	c.innerClient = mgr.GetClient()
	err = ctrl.NewControllerManagedBy(mgr).For(&flowv1.PriorityLevelConfiguration{}).Complete(reconcile.Func(c.ReconcilePLC))
	if err != nil {
		panic(err)
	}

	go func() {
		err = mgr.Start(context.Background())
		if err != nil {
			panic(err)
		}
	}()
}

type controller struct {
	outerClient client.Client
	innerClient client.Client
	mut         sync.Mutex

	// Counter state
	lastSamples map[types.NamespacedName]time.Time // NOTE: need to prune these when pods are deleted
	lastStates  map[types.NamespacedName]corev1.ConditionStatus
	counter     float64
	lastCounter float64

	// PID controller state
	integral   float64
	lastError  float64
	lastUpdate time.Time
}

func (c *controller) ReconcilePod(ctx context.Context, req reconcile.Request) (reconcile.Result, error) {
	logger := ctrl.LoggerFrom(ctx)

	pod := &corev1.Pod{}
	err := c.outerClient.Get(ctx, req.NamespacedName, pod)
	if err != nil {
		return reconcile.Result{}, client.IgnoreNotFound(err)
	}
	if !strings.HasPrefix(pod.Name, "kube-apiserver") || pod.Status.Phase != corev1.PodRunning {
		return reconcile.Result{}, nil
	}

	c.mut.Lock()
	defer c.mut.Unlock()

	const podStartupGracePeriod = time.Minute
	for _, cond := range pod.Status.Conditions {
		// NOTE: there should probably be a pod startup graceperiod

		if cond.Type != corev1.ContainersReady {
			continue
		}

		// Retain the last observed state
		lastState, ok := c.lastStates[req.NamespacedName]
		if !ok {
			logger.Info("first sample", "pod", pod.Name, "state", cond.Status)
		}
		c.lastStates[req.NamespacedName] = cond.Status

		// Retain the time delta between samples
		currentSample := time.Now()
		lastSample, ok := c.lastSamples[req.NamespacedName]
		c.lastSamples[req.NamespacedName] = currentSample
		if !ok {
			// First sample - no signal yet
			return reconcile.Result{RequeueAfter: time.Second}, nil
		}

		// Count the entire duration since the last sample as non-ready if the condition has been false the whole time
		if lastState == corev1.ConditionFalse && cond.Status == corev1.ConditionFalse {
			seconds := currentSample.Sub(lastSample).Seconds()
			// logger.Info("incrementing counter", "reason", "StillFalse", "amount", seconds, "pod", pod.Name)
			c.counter += seconds
		}

		// Count only the time since the last transition to false if the condition was true at the last sample
		if lastState == corev1.ConditionTrue && cond.Status == corev1.ConditionFalse {
			seconds := currentSample.Sub(cond.LastTransitionTime.Time).Seconds()
			// logger.Info("incrementing counter", "reason", "BecameFalse", "amount", seconds, "pod", pod.Name)
			c.counter += seconds
		}

		// Count only the time from the last sample until the transition time if the condition has since transitioned to true
		if lastState == corev1.ConditionFalse && cond.Status == corev1.ConditionTrue {
			seconds := lastSample.Sub(cond.LastTransitionTime.Time).Seconds()
			// logger.Info("incrementing counter", "reason", "BecameTrue", "amount", seconds, "pod", pod.Name)
			c.counter += seconds
		}

		if cond.Status == corev1.ConditionFalse {
			return reconcile.Result{RequeueAfter: time.Second}, nil
		}
		break
	}

	return reconcile.Result{}, nil
}

func (c *controller) ReconcilePLC(ctx context.Context, req reconcile.Request) (reconcile.Result, error) {
	logger := ctrl.LoggerFrom(ctx)

	plc := &flowv1.PriorityLevelConfiguration{}
	err := c.innerClient.Get(ctx, req.NamespacedName, plc)
	if err != nil {
		return reconcile.Result{}, client.IgnoreNotFound(err)
	}
	if plc.Name != "safety-limit" { // NOTE: could use a field selector on the list/watch instead
		return reconcile.Result{}, nil
	}

	const interval = time.Second * 5
	old := ptr.Deref(plc.Spec.Limited.NominalConcurrencyShares, 0)
	val, ok := c.pid(old)
	if !ok {
		return reconcile.Result{RequeueAfter: interval}, nil
	}

	copy := plc.DeepCopy()
	copy.Spec.Limited.NominalConcurrencyShares = &val
	err = c.innerClient.Patch(ctx, copy, client.StrategicMergeFrom(plc))
	if err != nil {
		logger.Error(err, "error updating concurrency limit", "value", val)
		// NOTE: needs to be more aggressive than the default retry timings, but should still probably use exponential back off
		return reconcile.Result{RequeueAfter: time.Millisecond * 50}, nil
	}

	logger.Info("updated concurrency limit", "value", val, "old", old)

	return reconcile.Result{RequeueAfter: interval}, nil
}

func (c *controller) pid(current int32) (newShares int32, ok bool) {
	now := time.Now()

	c.mut.Lock()
	defer c.mut.Unlock()

	if c.lastUpdate.IsZero() {
		c.lastUpdate = now
		c.lastError = 0
		c.integral = 0
		return 0, false
	}

	dt := now.Sub(c.lastUpdate).Seconds()
	if dt < 0.1 {
		return 0, false // too soon
	}

	counterDelta := c.counter - c.lastCounter
	c.lastCounter = c.counter
	pidError := 0 - float64(counterDelta)
	c.integral += pidError * dt // NOTE: this should be persisted in a cr or configmap in case this process crashes

	// add a small positive bias to the integral to slowly raise the limit during recovery
	if counterDelta <= 0 {
		c.integral += 3 * dt
	}

	// Clamp the integral term
	if c.integral > maxIntegral {
		c.integral = maxIntegral
	} else if c.integral < -maxIntegral {
		c.integral = -maxIntegral
	}

	derivative := (pidError - c.lastError) / dt
	controlOutput := kP*pidError + kI*c.integral + kD*derivative

	// clamp
	raw := float64(current) + controlOutput
	if raw < float64(minShares) {
		raw = float64(minShares)
	} else if raw > float64(maxShares) {
		raw = float64(maxShares)
	}
	newShares = int32(raw)

	c.lastError = pidError
	c.lastUpdate = now
	return newShares, true
}
