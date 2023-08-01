//date: 2023-08-01T16:48:44Z
//url: https://api.github.com/gists/c8cbd1825c1a7cdaed81aed9336c97ac
//owner: https://api.github.com/users/xrstf

package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"regexp"
	"strings"

	kcptenancyv1alpha1 "github.com/kcp-dev/kcp/sdk/apis/tenancy/v1alpha1"
	"github.com/kcp-dev/logicalcluster/v3"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/klog/v2"
	ctrlruntimeclient "sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/cluster"
	"sigs.k8s.io/controller-runtime/pkg/kcp"
	"sigs.k8s.io/controller-runtime/pkg/kontext"
)

func main() {
	klog.InitFlags(nil)
	flag.Parse()

	if err := kcptenancyv1alpha1.AddToScheme(scheme.Scheme); err != nil {
		log.Fatalf("Failed to register kcptenancyv1alpha1 scheme: %v", err)
	}

	kcpConfig, err := clientcmd.LoadFromFile("kubeconfig")
	if err != nil {
		log.Fatalf("Failed to build kubeconfig for kcp cluster: %v", err)
	}

	restConfig, err := clientcmd.NewDefaultClientConfig(*kcpConfig, nil).ClientConfig()
	if err != nil {
		log.Fatalf("Failed to REST config for kcp cluster: %v", err)
	}

	usePatchedImplementation := true

	var reader ctrlruntimeclient.Reader
	if usePatchedImplementation {
		reader, err = NewClusterAwareAPIReader(restConfig, ctrlruntimeclient.Options{})
	} else {
		reader, err = kcp.NewClusterAwareAPIReader(restConfig, ctrlruntimeclient.Options{})
	}

	if err != nil {
		log.Fatalf("Failed to connect to kcp cluster: %v", err)
	}

	listWorkspaces(kontext.WithCluster(context.Background(), "root"), reader, 0)
}

func listWorkspaces(ctx context.Context, reader ctrlruntimeclient.Reader, indent int) {
	clusterName, _ := kontext.ClusterFrom(ctx)
	clusterPath := clusterName.Path()
	prefix := strings.Repeat(" ", indent)

	workspaceList := kcptenancyv1alpha1.WorkspaceList{}
	if err := reader.List(ctx, &workspaceList); err != nil {
		fmt.Printf(fmt.Sprintf("%s! failed to list workspaces: %v\n", prefix, err))
		return
	}

	for _, workspace := range workspaceList.Items {
		fmt.Printf(fmt.Sprintf("%s* %s (%s, %s)\n", prefix, workspace.Name, workspace.Spec.Type.Name, workspace.Status.Phase))

		childPath := clusterPath.Join(workspace.Name)
		childName := logicalcluster.Name(childPath.String())

		listWorkspaces(kontext.WithCluster(ctx, childName), reader, indent+2)
	}
}

// ###################################################################################
// code copied from ctrl-runtime in order to patch the delegated client
// ###################################################################################

func NewClusterAwareAPIReader(config *rest.Config, opts ctrlruntimeclient.Options) (ctrlruntimeclient.Reader, error) {
	httpClient, err := ClusterAwareHTTPClient(config)
	if err != nil {
		return nil, err
	}
	opts.HTTPClient = httpClient
	return cluster.DefaultNewAPIReader(config, opts)
}

// ClusterAwareHTTPClient returns an http.Client with a cluster aware round tripper.
func ClusterAwareHTTPClient(config *rest.Config) (*http.Client, error) {
	httpClient, err := rest.HTTPClientFor(config)
	if err != nil {
		return nil, err
	}

	httpClient.Transport = newClusterRoundTripper(httpClient.Transport)
	return httpClient, nil
}

// clusterRoundTripper is a cluster aware wrapper around http.RoundTripper.
type clusterRoundTripper struct {
	delegate http.RoundTripper
}

// newClusterRoundTripper creates a new cluster aware round tripper.
func newClusterRoundTripper(delegate http.RoundTripper) *clusterRoundTripper {
	return &clusterRoundTripper{
		delegate: delegate,
	}
}

func (c *clusterRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	cluster, ok := kontext.ClusterFrom(req.Context())
	if ok {
		req = req.Clone(req.Context())
		req.URL.Path = generatePath(req.URL.Path, cluster.Path())
		req.URL.RawPath = generatePath(req.URL.RawPath, cluster.Path())
	}
	return c.delegate.RoundTrip(req)
}

// apiRegex matches any string that has /api/ or /apis/ in it.
var apiRegex = regexp.MustCompile(`(/api/|/apis/)`)

var clustersRegex = regexp.MustCompile(`^/clusters/[^/]+`)

// generatePath formats the request path to target the specified cluster.
func generatePath(originalPath string, clusterPath logicalcluster.Path) string {
	// HACK: strip any pre-existing /clusters/.... prefix
	originalPath = clustersRegex.ReplaceAllString(originalPath, "")

	// If the originalPath already has cluster.Path() then the path was already modifed and no change needed
	if strings.Contains(originalPath, clusterPath.RequestPath()) {
		return originalPath
	}
	// If the originalPath has /api/ or /apis/ in it, it might be anywhere in the path, so we use a regex to find and
	// replaces /api/ or /apis/ with $cluster/api/ or $cluster/apis/
	if apiRegex.MatchString(originalPath) {
		return apiRegex.ReplaceAllString(originalPath, fmt.Sprintf("%s$1", clusterPath.RequestPath()))
	}
	// Otherwise, we're just prepending /clusters/$name
	path := clusterPath.RequestPath()
	// if the original path is relative, add a / separator
	if len(originalPath) > 0 && originalPath[0] != '/' {
		path += "/"
	}
	// finally append the original path
	path += originalPath
	return path
}
