//date: 2024-04-23T16:54:41Z
//url: https://api.github.com/gists/d67c685a508976924ac7df2a51dbcc0e
//owner: https://api.github.com/users/dynac01

import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.cdimascio.dotenv.Dotenv;
import models.FedoraJsonResult;
import models.PidResult;
import models.TotalPidsResult;
import org.apache.http.auth.UsernamePasswordCredentials;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.auth.BasicScheme;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.select.Elements;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class Main {
    private final static Dotenv dotenv = Dotenv.load();
    private static final int BATCH_LIMIT = 1200;
    private static final ExecutorService executor = Executors.newFixedThreadPool(10);
    private static final ObjectMapper mapper = new ObjectMapper();
    private static String USERNAME;
    private static String PASSWORD;

    public static void main(String[] args) throws Exception {
        USERNAME = dotenv.get("USERNAME");
        PASSWORD = "**********"

        int totalPids = fetchTotalNumberOfPids();
        AtomicInteger totalFileSize = new AtomicInteger();

        for (int offset = 0; offset < totalPids; offset += BATCH_LIMIT) {
            List<PidResult> pids = fetchObjectPids(offset);
            System.out.println("Processing batch starting at offset: " + offset);

            for (PidResult pid : pids) {
                executor.submit(() -> {
                    try {
                        int fileSize = fetchAndStoreDatastreamSize(pid);
                        totalFileSize.addAndGet(fileSize);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                });
            }

            System.out.println("Batch complete. Total File Size so far: " + totalFileSize.get() + " bytes");
        }

        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.HOURS);

        System.out.println("Total File Size: " + totalFileSize + " bytes");
        System.out.println("Average File Size: " + (totalFileSize.get() / totalPids) + " bytes");
    }

    /**
     * Get the total number of PIDs in the fedora instance
     *
     * @return total number of PID
     * @throws Exception
     */
    private static int fetchTotalNumberOfPids() throws Exception {
        System.out.println("Fetching total number of PIDs");
        final String url = dotenv.get("URL") + ":8080/fedora/risearch";
        CloseableHttpClient client = HttpClients.createDefault();
        HttpPost post = new HttpPost(url);
        post.setHeader("Content-Type", "application/x-www-form-urlencoded");
        post.setHeader("Accept", "application/json");
        post.setEntity(
                new StringEntity(
                        "type=tuples&lang=sparql&format=json&query=" +
                                "PREFIX fedora: <info:fedora/fedora-system:def/model#> " +
                                "SELECT (COUNT(DISTINCT ?s) AS ?totalPIDs) " +
                                "WHERE { ?s fedora:hasModel ?o . }"
                )
        );

        UsernamePasswordCredentials creds = "**********"
        post.addHeader(new BasicScheme().authenticate(creds, post, null));

        CloseableHttpResponse response = client.execute(post);
        String jsonResponse = EntityUtils.toString(response.getEntity());
        response.close();
        client.close();

        FedoraJsonResult<TotalPidsResult> result = mapper
                .readValue(
                        jsonResponse,
                        mapper.getTypeFactory().constructParametricType(FedoraJsonResult.class, TotalPidsResult.class)
                );

        int totalNumberOfPids = Integer.parseInt(result.getResults().getFirst().getTotalPIDs());
        System.out.println("Total number of PIDs is " + totalNumberOfPids);
        return totalNumberOfPids;
    }

    /**
     * Fetches all PIDs in the instance including corresponding datastream IDs
     *
     * @param offset Current batch offset
     * @return A list of PidResults
     * @throws Exception
     */
    private static List<PidResult> fetchObjectPids(int offset) throws Exception {
        System.out.println("Fetching object PIDs");
        final String url = dotenv.get("URL") + ":8080/fedora/risearch";
        CloseableHttpClient client = HttpClients.createDefault();
        HttpPost post = new HttpPost(url);
        post.setHeader("Content-Type", "application/x-www-form-urlencoded");
        post.setHeader("Accept", "application/json");
        post.setEntity(new StringEntity(
                "type=tuples&lang=sparql&format=json&limit=" + BATCH_LIMIT + "&offset=" + offset +
                        "&query=PREFIX fedora: <info:fedora/fedora-system:def/model#> " +
                        "PREFIX view: <info:fedora/fedora-system:def/view#> " +
                        "SELECT ?pid ?dsid " +
                        "WHERE { " +
                        "?s fedora:hasModel ?o . " +
                        "?s view:disseminates ?ds . " +
                        "BIND(REPLACE(STR(?s), \"info:fedora/\", \"\") AS ?pid) " +
                        "BIND(REPLACE(STR(?ds), \"info:fedora/\", \"\") AS ?dsid) " +
                        "}"));

        UsernamePasswordCredentials creds = "**********"
        post.addHeader(new BasicScheme().authenticate(creds, post, null));

        CloseableHttpResponse response = client.execute(post);
        String jsonResponse = EntityUtils.toString(response.getEntity());
        response.close();
        client.close();

        FedoraJsonResult<PidResult> result = mapper
                .readValue(
                        jsonResponse,
                        mapper.getTypeFactory().constructParametricType(FedoraJsonResult.class, PidResult.class)
                );

        return result.getResults();
    }

    /**
     * Gets the file size for an individual PID
     *
     * @param pidResult The current PID for which we are getting the file size
     * @return The file size in bytes
     * @throws Exception
     */
    private static int fetchAndStoreDatastreamSize(PidResult pidResult) throws Exception {
        final String url =
                dotenv.get("URL")
                        + ":8080/fedora/objects/"
                        + pidResult.getPid()
                        + "/datastreams/"
                        + Arrays.stream(pidResult.getDsid().split("/")).toList().get(1);

        CloseableHttpClient client = HttpClients.createDefault();
        HttpGet get = new HttpGet(url);

        UsernamePasswordCredentials creds = "**********"
        get.addHeader(new BasicScheme().authenticate(creds, get, null));

        CloseableHttpResponse response = client.execute(get);
        String responseContent = EntityUtils.toString(response.getEntity());
        Document doc = Jsoup.parse(responseContent);
        Elements sizeElements = doc.select("td:contains(Datastream Size) + td");
        response.close();
        client.close();

        return Integer.parseInt(Objects.requireNonNull(sizeElements.first()).text().trim());
    }

}
        client.close();

        return Integer.parseInt(Objects.requireNonNull(sizeElements.first()).text().trim());
    }

}
