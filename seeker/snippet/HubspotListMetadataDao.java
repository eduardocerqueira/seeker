//date: 2025-03-19T17:14:16Z
//url: https://api.github.com/gists/f849b6d0e9a4fbdd443b79c6eb414de5
//owner: https://api.github.com/users/orren-at-webflow

package com.intellimize.integration.hubspot.dao;

import static com.google.common.base.Preconditions.checkState;
import static com.intellimize.protobuf.IntegrationDataStateProtos.HubspotObjectType.HUBSPOT_OBJECT_TYPE_COMPANY_LIST;
import static com.intellimize.protobuf.IntegrationDataStateProtos.HubspotObjectType.HUBSPOT_OBJECT_TYPE_CONTACT_LIST;
import static com.intellimize.protobuf.IntegrationDataStateProtos.HubspotObjectType.HUBSPOT_OBJECT_TYPE_DEAL_LIST;
import static java.util.stream.Collectors.toList;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.gson.Gson;
import com.intellimize.integration.hubspot.dao.HubspotListMetadataResponse.HubspotListMetadata;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpRequest.BodyPublishers;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import com.intellimize.integration.util.ServiceProperties;
import com.intellimize.protobuf.IntegrationDataStateProtos.HubspotObjectType;
import org.apache.http.HttpHeaders;

public class HubspotListMetadataDao {
  private static final Gson gson = new Gson();
  // The list search API has a higher limit than other endpoints. This is a limit enforced by Hubspot.
  private static final int HUBSPOT_PER_REQUEST_LIMIT = 500;
  private static final ServiceProperties properties = ServiceProperties.getInstance();

  // These are limits enforced by us, not to be confused with Hubspot limit above.
  private static final class OptimizeListLimits {
    private final int totalListsLimit;
    private final int listSizeLimit;

    private OptimizeListLimits(int totalListsLimit, int listSizeLimit) {
      this.totalListsLimit = totalListsLimit;
      this.listSizeLimit = listSizeLimit;
    }

    public int getTotalListsLimit() {
      return totalListsLimit;
    }

    public int getListSizeLimit() {
      return listSizeLimit;
    }
  }
  
  private static final Map<HubspotObjectType, OptimizeListLimits> OBJECT_TYPE_LIMITS =
      ImmutableMap.<HubspotObjectType, OptimizeListLimits>builder()
          .put(HUBSPOT_OBJECT_TYPE_CONTACT_LIST, 
              new OptimizeListLimits(
                  properties.getHubspotContactListTotalRecordLimit(),
                  properties.getHubspotContactListSizeLimit()))
          .put(HUBSPOT_OBJECT_TYPE_COMPANY_LIST, 
              new OptimizeListLimits(
                  properties.getHubspotCompanyListTotalRecordLimit(),
                  properties.getHubspotCompanyListSizeLimit()))
          .put(HUBSPOT_OBJECT_TYPE_DEAL_LIST, 
              new OptimizeListLimits(
                  properties.getHubspotDealListTotalRecordLimit(),
                  properties.getHubspotDealListSizeLimit()))
          .build();

  private final HttpClient client;
  private final HubspotTokenDao hubspotTokenDao;

  public HubspotListMetadataDao(HttpClient client, HubspotTokenDao hubspotTokenDao) {
    this.client = client;
    this.hubspotTokenDao = "**********"
  }

  public List<HubspotListMetadata> getListsOfType(HubspotObjectType objectType) {
    return getAllLists().stream()
        .filter(list -> list.getObjectType() == objectType)
        .filter(list -> list.getAdditionalProperties().getListSize() < OBJECT_TYPE_LIMITS.get(objectType).getListSizeLimit())
        .sorted(Comparator.comparing(
            list -> {
              Instant updatedAt = Instant.parse(list.getUpdatedAt());
              Optional<HubspotListMetadataResponse.AdditionalProperties> additionalProperties =
                  Optional.ofNullable(list.getAdditionalProperties());
              Instant lastRecordAdded = additionalProperties
                  .map(HubspotListMetadataResponse.AdditionalProperties::getLastRecordAddedAt)
                  .map(timestamp -> Instant.ofEpochMilli(Long.parseLong(timestamp)))
                  .orElse(Instant.EPOCH);
              Instant lastRecordRemoved = additionalProperties
                  .map(HubspotListMetadataResponse.AdditionalProperties::getLastRecordRemovedAt)
                  .map(timestamp -> Instant.ofEpochMilli(Long.parseLong(timestamp)))
                  .orElse(Instant.EPOCH);
              return Collections.max(ImmutableList.of(updatedAt, lastRecordAdded, lastRecordRemoved));
            }, Comparator.reverseOrder()))
        .limit(OBJECT_TYPE_LIMITS.get(objectType).getTotalListsLimit())
        .collect(toList());
  }


  private List<HubspotListMetadata> getAllLists() {
    List<HubspotListMetadata> results = new ArrayList<>();
    int offset = 0;
    boolean hasMore;
    do {
      HubspotListMetadataResponse response = "**********"
      results.addAll(response.getLists());
      int newOffset = response.getOffset();
      checkState(
          response.getLists().isEmpty() || offset < newOffset,
          "Expected new offset to be greater than current offset %s. Found %s",
          offset,
          newOffset);
      offset = newOffset;
      hasMore = response.getHasMore();
    } while (hasMore);
    return results;
  }

  private static HubspotListMetadataResponse getListMetadataBatch(
          HttpClient httpClient,
          HubspotTokenDao hubspotTokenDao,
          int offset) {

    Map<String, Object> request =
        ImmutableMap.<String, Object>builder()
                .put("count", HUBSPOT_PER_REQUEST_LIMIT)
                .put("offset", offset)
                .build();
    String body = gson.toJson(request);
    HttpRequest.Builder builder =
        HttpRequest.newBuilder()
            .uri(URI.create("https://api.hubapi.com/crm/v3/lists/search"))
            .header(HttpHeaders.CONTENT_TYPE, "application/json")
            .POST(BodyPublishers.ofString(body));
    return HubspotUtil.sendRequest(
        builder, httpClient, hubspotTokenDao, HubspotListMetadataResponse.class);
  }
}
ao, HubspotListMetadataResponse.class);
  }
}
