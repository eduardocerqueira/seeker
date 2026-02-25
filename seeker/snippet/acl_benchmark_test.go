//date: 2026-02-25T17:35:37Z
//url: https://api.github.com/gists/8ce13667a3ca3ecb1d5b57e038ba5f69
//owner: https://api.github.com/users/trozet

package ops

import (
	"context"
	"fmt"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	libovsdbclient "github.com/ovn-kubernetes/libovsdb/client"
	"github.com/ovn-kubernetes/libovsdb/model"
	"github.com/ovn-kubernetes/libovsdb/ovsdb"

	"github.com/ovn-org/ovn-kubernetes/go-controller/pkg/nbdb"
	libovsdbtest "github.com/ovn-org/ovn-kubernetes/go-controller/pkg/testing/libovsdb"
	"github.com/ovn-org/ovn-kubernetes/go-controller/pkg/types"
)

const (
	aclUpdatePoolSize       = 2048
	aclContentionFanout     = 16
	aclContentionMaxRetries = 5
	aclBenchPrimaryIDKey    = "bench-acl"
	aclBenchPortGroupName   = "bench-acl-portgroup"
	aclBenchDupErrContains  = "multiple results"
)

/*
/home/trozet/.cache/JetBrains/GoLand2024.2/tmp/GoLand/___BenchmarkACLCreateUnique_in_github_com_ovn_org_ovn_kubernetes_go_controller_pkg_libovsdb_ops.test -test.v -test.paniconexit0 -test.bench ^\QBenchmarkACLCreateUnique\E$ -test.run ^$
goos: linux
goarch: amd64
pkg: github.com/ovn-org/ovn-kubernetes/go-controller/pkg/libovsdb/ops
cpu: Intel(R) Core(TM) Ultra 9 285K
BenchmarkACLCreateUnique
BenchmarkACLCreateUnique/NoGuard
BenchmarkACLCreateUnique/NoGuard-24         	    1542	    659940 ns/op	         0 errors	         0 guard_ops
BenchmarkACLCreateUnique/WithGuard
BenchmarkACLCreateUnique/WithGuard-24       	    1676	   1044021 ns/op	         0 errors	      1676 guard_ops
PASS

Process finished with the exit code 0
*/
func BenchmarkACLCreateUnique(b *testing.B) {
	for _, useGuard := range []bool{false, true} {
		name := "NoGuard"
		if useGuard {
			name = "WithGuard"
		}
		b.Run(name, func(b *testing.B) {
			nbClient := newACLBenchNBClient(b)
			var errs uint64
			var guardOps uint64

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				pid := fmt.Sprintf("%s-create-%d", aclBenchPrimaryIDKey, i)
				acl := newBenchmarkACL(pid, uint64(i))
				guardApplied, err := createOrUpdateACLForBenchmark(nbClient, acl, useGuard)
				if guardApplied {
					atomic.AddUint64(&guardOps, 1)
				}
				if err != nil {
					atomic.AddUint64(&errs, 1)
				}
			}
			b.StopTimer()
			b.ReportMetric(float64(errs), "errors")
			b.ReportMetric(float64(guardOps), "guard_ops")
		})
	}
}

/*
/home/trozet/.cache/JetBrains/GoLand2024.2/tmp/GoLand/___BenchmarkACLUpdateExisting_in_github_com_ovn_org_ovn_kubernetes_go_controller_pkg_libovsdb_ops.test -test.v -test.paniconexit0 -test.bench ^\QBenchmarkACLUpdateExisting\E$ -test.run ^$
goos: linux
goarch: amd64
pkg: github.com/ovn-org/ovn-kubernetes/go-controller/pkg/libovsdb/ops
cpu: Intel(R) Core(TM) Ultra 9 285K
BenchmarkACLUpdateExisting
BenchmarkACLUpdateExisting/NoGuard
BenchmarkACLUpdateExisting/NoGuard-24         	    2948	    550571 ns/op	         0 errors	         0 guard_ops
BenchmarkACLUpdateExisting/WithGuard
BenchmarkACLUpdateExisting/WithGuard-24       	    1880	    553009 ns/op	         0 errors	         0 guard_ops
PASS

Process finished with the exit code 0
*/
func BenchmarkACLUpdateExisting(b *testing.B) {
	for _, useGuard := range []bool{false, true} {
		name := "NoGuard"
		if useGuard {
			name = "WithGuard"
		}
		b.Run(name, func(b *testing.B) {
			nbClient := newACLBenchNBClient(b)
			ids := make([]string, 0, aclUpdatePoolSize)
			uuids := make([]string, 0, aclUpdatePoolSize)
			for i := 0; i < aclUpdatePoolSize; i++ {
				pid := fmt.Sprintf("%s-update-%d", aclBenchPrimaryIDKey, i)
				uuid := fmt.Sprintf("00000000-0000-0000-0000-%012d", i+1)
				ids = append(ids, pid)
				uuids = append(uuids, uuid)
				acl := newBenchmarkACL(pid, uint64(i))
				acl.UUID = uuid
				_, err := createOrUpdateACLForBenchmark(nbClient, acl, false)
				if err != nil {
					b.Fatalf("setup create failed: %v", err)
				}
			}
			if err := waitForACLCount(nbClient, aclUpdatePoolSize); err != nil {
				b.Fatalf("setup cache sync failed: %v", err)
			}

			var errs uint64
			var guardOps uint64
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				idx := i % len(ids)
				pid := ids[idx]
				acl := newBenchmarkACL(pid, uint64(i+aclUpdatePoolSize))
				acl.UUID = uuids[idx]
				guardApplied, err := createOrUpdateACLForBenchmark(nbClient, acl, useGuard)
				if guardApplied {
					atomic.AddUint64(&guardOps, 1)
				}
				if err != nil {
					atomic.AddUint64(&errs, 1)
				}
			}
			b.StopTimer()
			b.ReportMetric(float64(errs), "errors")
			b.ReportMetric(float64(guardOps), "guard_ops")
		})
	}
}

/*
/home/trozet/.cache/JetBrains/GoLand2024.2/tmp/GoLand/___BenchmarkACLCreateOrUpdateContention_in_github_com_ovn_org_ovn_kubernetes_go_controller_pkg_libovsdb_ops.test -test.v -test.paniconexit0 -test.bench ^\QBenchmarkACLCreateOrUpdateContention\E$ -test.run ^$
goos: linux
goarch: amd64
pkg: github.com/ovn-org/ovn-kubernetes/go-controller/pkg/libovsdb/ops
cpu: Intel(R) Core(TM) Ultra 9 285K
BenchmarkACLCreateOrUpdateContention
BenchmarkACLCreateOrUpdateContention/NoGuard
BenchmarkACLCreateOrUpdateContention/NoGuard-24         	    2420	    515592 ns/op	         0 dup_errors	         0 errors	         0 guard_ops
BenchmarkACLCreateOrUpdateContention/WithGuard
BenchmarkACLCreateOrUpdateContention/WithGuard-24       	    3901	    323163 ns/op	         0 dup_errors	         0 errors	      3024 guard_ops
PASS

Process finished with the exit code 0
*/
func BenchmarkACLCreateOrUpdateContention(b *testing.B) {
	for _, useGuard := range []bool{false, true} {
		name := "NoGuard"
		if useGuard {
			name = "WithGuard"
		}
		b.Run(name, func(b *testing.B) {
			nbClient := newACLBenchNBClient(b)
			var seq uint64
			var errs uint64
			var dupErrs uint64
			var guardOps uint64

			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					n := atomic.AddUint64(&seq, 1) - 1
					pid := fmt.Sprintf("%s-contention-%d", aclBenchPrimaryIDKey, n/aclContentionFanout)
					acl := newBenchmarkACL(pid, n)
					var err error
					for attempt := 0; attempt < aclContentionMaxRetries; attempt++ {
						var guardApplied bool
						guardApplied, err = createOrUpdateACLForBenchmark(nbClient, acl, useGuard)
						if guardApplied {
							atomic.AddUint64(&guardOps, 1)
						}
						if err == nil {
							break
						}
						if strings.Contains(err.Error(), aclBenchDupErrContains) {
							atomic.AddUint64(&dupErrs, 1)
						}
						runtime.Gosched()
					}
					if err != nil {
						atomic.AddUint64(&errs, 1)
					}
				}
			})
			b.StopTimer()
			b.ReportMetric(float64(errs), "errors")
			b.ReportMetric(float64(dupErrs), "dup_errors")
			b.ReportMetric(float64(guardOps), "guard_ops")
		})
	}
}

func newACLBenchNBClient(b *testing.B) libovsdbclient.Client {
	b.Helper()
	initialData := []libovsdbtest.TestData{
		&nbdb.PortGroup{
			Name: aclBenchPortGroupName,
		},
	}
	setup := libovsdbtest.TestSetup{
		NBData: initialData,
	}
	nbClient, cleanup, err := libovsdbtest.NewNBTestHarness(setup, nil)
	if err != nil {
		b.Fatalf("failed to create NB test harness: %v", err)
	}
	b.Cleanup(cleanup.Cleanup)
	return nbClient
}

func createOrUpdateACLForBenchmark(nbClient libovsdbclient.Client, acl *nbdb.ACL, useGuard bool) (bool, error) {
	ops, err := CreateOrUpdateACLsOps(nbClient, nil, nil, acl)
	if err != nil {
		return false, err
	}
	guardApplied := false
	if useGuard && aclInsertOpPresent(ops) {
		guardOps, err := buildACLWaitGuardOps(nbClient, acl)
		if err != nil {
			return false, err
		}
		ops = append(guardOps, ops...)
		guardApplied = true
	}
	ops, err = AddACLsToPortGroupOps(nbClient, ops, aclBenchPortGroupName, acl)
	if err != nil {
		return guardApplied, err
	}
	_, err = TransactAndCheck(nbClient, ops)
	return guardApplied, err
}

func waitForACLCount(nbClient libovsdbclient.Client, count int) error {
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		acls := []*nbdb.ACL{}
		err := nbClient.WhereCache(func(*nbdb.ACL) bool { return true }).List(ctx, &acls)
		cancel()
		if err == nil && len(acls) >= count {
			return nil
		}
		time.Sleep(20 * time.Millisecond)
	}
	return fmt.Errorf("timed out waiting for %d ACLs in cache", count)
}

func aclInsertOpPresent(ops []ovsdb.Operation) bool {
	for i := range ops {
		if ops[i].Op == ovsdb.OperationInsert && ops[i].Table == nbdb.ACLTable {
			return true
		}
	}
	return false
}

func buildACLWaitGuardOps(nbClient libovsdbclient.Client, acl *nbdb.ACL) ([]ovsdb.Operation, error) {
	timeout := types.OVSDBWaitTimeout
	primaryID, ok := acl.ExternalIDs[types.PrimaryIDKey]
	if !ok || primaryID == "" {
		return nil, fmt.Errorf("missing ACL primary ID")
	}

	waitACL := &nbdb.ACL{
		ExternalIDs: map[string]string{types.PrimaryIDKey: primaryID},
	}
	cond := model.Condition{
		Field:    &waitACL.ExternalIDs,
		Function: ovsdb.ConditionIncludes,
		Value: map[string]string{
			types.PrimaryIDKey: primaryID,
		},
	}
	return nbClient.WhereAny(waitACL, cond).Wait(ovsdb.WaitConditionNotEqual, &timeout, waitACL, &waitACL.ExternalIDs)
}

func newBenchmarkACL(primaryID string, nonce uint64) *nbdb.ACL {
	return &nbdb.ACL{
		Action:    nbdb.ACLActionAllowRelated,
		Direction: nbdb.ACLDirectionToLport,
		ExternalIDs: map[string]string{
			types.PrimaryIDKey: primaryID,
		},
		Log:      false,
		Match:    fmt.Sprintf("ip4.src == 10.%d.%d.%d", (nonce>>16)%256, (nonce>>8)%256, nonce%256),
		Options:  map[string]string{},
		Priority: types.DefaultAllowPriority,
		Tier:     types.DefaultACLTier,
	}
}
