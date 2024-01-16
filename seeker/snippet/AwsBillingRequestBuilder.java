//date: 2024-01-16T16:51:17Z
//url: https://api.github.com/gists/68634e30d8a683acb0ed66b6e52ceefc
//owner: https://api.github.com/users/hwanseok-dev

package io.whatap.account.web.service.metering.aws;

import com.amazonaws.services.marketplacemetering.model.BatchMeterUsageRequest;
import com.amazonaws.services.marketplacemetering.model.Tag;
import com.amazonaws.services.marketplacemetering.model.UsageAllocation;
import com.amazonaws.services.marketplacemetering.model.UsageRecord;
import io.whatap.lang.ref.INT;
import io.whatap.util.StringEnumer;
import io.whatap.util.StringKeyLinkedMap;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.stream.Collectors;

public class AwsBillingRequestBuilder {
    /**
     * AWS MP 제품 별 총 과금량
     *
     * 하나의 제품을 여러 개의 계정이 구독할 수 있음
     */
    private final StringKeyLinkedMap<List<AccountUsage>> productCodeMap = new StringKeyLinkedMap<List<AccountUsage>>() {
        @Override
        protected List<AccountUsage> create(String awsProductCode) {
            return new ArrayList<>();
        }
    };

    public AwsBillingRequestBuilder(){

    }

    public void add(AccountUsage accountUsage) {
        this.productCodeMap.intern(accountUsage.awsProductCode).add(accountUsage);
    }

    public List<BatchMeterUsageRequest> build(){
        List<BatchMeterUsageRequest> allProductRequests = new ArrayList<>();
        StringEnumer keyEn = productCodeMap.keys();
        while (keyEn.hasMoreElements()) {
            // AWS MP 제품 코드 조회
            String awsProductCode = keyEn.nextString();

            // AWS MP 제품을 구독중인 계정 목록 조회
            List<AccountUsage> accountUsages = productCodeMap.get(awsProductCode);

            // AWS MP 제품에 포함된 미터링 정보를 최대 25개씩 자름
            List<List<UsageRecord>> usageRecordsUpTo25 = splitUsageRecordsUpTo25(accountUsages.stream()
                    .map(AccountUsage::toUsageRecord)
                    .flatMap(List::stream)
                    .collect(Collectors.toList()));

            // 잘라진 미터링 정보를 요청 정보로 변경
            List<BatchMeterUsageRequest> oneProductRequests = usageRecordsUpTo25.stream()
                    .map(usageRecords -> new BatchMeterUsageRequest()
                            .withProductCode(awsProductCode)
                            .withUsageRecords(usageRecords))
                    .collect(Collectors.toList());

            allProductRequests.addAll(oneProductRequests);
        }
        return allProductRequests;
    }

    /**
     * BatchMeterUsage can process up to 25 UsageRecords at a time.
     *
     * The maximum payload size can't be more than 1 MB.
     * This includes input attribute keys (for example, UsageRecords, AllocatedUsageQuantity, tags).
     */
    private List<List<UsageRecord>> splitUsageRecordsUpTo25(List<UsageRecord> usageRecords) {
        List<List<UsageRecord>> result = new ArrayList<>();
        for (int i = 0; i < usageRecords.size(); i += 25) {
            int end = Math.min(i + 25, usageRecords.size());
            List<UsageRecord> sublist = usageRecords.subList(i, end);
            result.add(sublist);
        }
        return result;
    }


    /**
     * 계정 별 사용량
     *
     * 하나의 계정에서 여러 개의 dimension(ProductType)를 사용할 수 있고
     * 각 dimension 별로 여러 개의 Project가 있을 수 있음
     */
    public static class AccountUsage {
        private final String awsProductCode;
        private final Date timestamp;
        private final String customerIdentifier;

        // dimension -  pcode 별 과금액
        private final StringKeyLinkedMap<List<ProjectUsage>> dimensionMap = new StringKeyLinkedMap<List<ProjectUsage>>() {
            @Override
            protected List<ProjectUsage> create(String key) {
                return new ArrayList<>();
            }
        };

        public AccountUsage(String awsProductCode, Date timestamp, String customerIdentifier) {
            this.awsProductCode = awsProductCode;
            this.timestamp = timestamp;
            this.customerIdentifier = customerIdentifier;
        }

        public void add(ProjectUsage projectUsage) {
            dimensionMap.intern(projectUsage.identifier).add(projectUsage);
        }

        public void addAll(List<ProjectUsage> projectUsages) {
            projectUsages.forEach(this::add);
        }

        /*
         * UsageRecord : 하나의 AWS MP 계정으로 구독한 dimension 별 총 사용량
         *   - timestamp
         *   - customerIdentifier
         *   - dimension : productType
         *   - quantity
         *   - List<UsageAllocation> : pcode 별 사용량
         *       - allocatedUsageQuantity
         *       - List<Tag>
         *           - key
         *           - value
         */
        public List<UsageRecord> toUsageRecord(){
            List<UsageRecord> allDimensionUsage = new ArrayList<>();
            StringEnumer keyEn = dimensionMap.keys();
            while (keyEn.hasMoreElements()) {
                String dimension = keyEn.nextString();

                /**
                 * The sum of AllocatedUsageQuantity of UsageAllocation must equal the UsageQuantity, which is the aggregate usage.
                 */
                INT totalQuantity = new INT();
                List<UsageAllocation> usageAllocations = dimensionMap.get(dimension)
                        .stream()
                        .peek(byPcode -> totalQuantity.value += byPcode.quantity)
                        .map(ProjectUsage::toUsageAllocation)
                        .collect(Collectors.toList());

                UsageRecord oneDimensionUsage = new UsageRecord()
                        .withTimestamp(timestamp)
                        .withCustomerIdentifier(customerIdentifier)
                        .withDimension(dimension)
                        .withUsageAllocations(usageAllocations)
                        .withQuantity(totalQuantity.value);
                allDimensionUsage.add(oneDimensionUsage);
            }
            return allDimensionUsage;
        }
    }

    /**
     * 프로젝트 별 사용량
     *
     * 프로젝트에 포함된 dimension(ProductType) 정보가 함께 저장되어야 dimension 별 합산이 가능하다
     *
     * | identifier                                           |
     * |------------------------------------------------------|
     * | server_host                                          |
     * | kubernetes_application_container                     |
     * | kubernetes_container                                 |
     * | database_host                                        |
     * | log                                                  |
     * | database_vcpu                                        |
     * | browser_session                                      |
     *
     */
    public static class ProjectUsage {
        /*
         * 1. dimension = API identifier + Description
         * 2. API identifier can have up to 60 characters
         * 3. API identifier consists of alphanumeric and underbar.
         */
        private final String identifier;
        private final long pcode;
        private final int quantity;

        /**
         * Maximum tags across UsageAllocation list – 5
         */
        private final List<Tag> tags;

        public static ProjectUsage forAppCore(long pcode, int quantity) {
            return new ProjectUsage("application_vcpu", pcode, quantity);
        }
        public static ProjectUsage forServerHost(long pcode, int quantity) {
            return new ProjectUsage("server_host", pcode, quantity);
        }
        public static ProjectUsage forK8sContainerWithApp(long pcode, int quantity) {
            return new ProjectUsage("kubernetes_application_container", pcode, quantity);
        }

        public static ProjectUsage forK8sContainers(long pcode, int quantity) {
            return new ProjectUsage("kubernetes_container", pcode, quantity);
        }
        public static ProjectUsage forDBHost(long pcode, int quantity) {
            return new ProjectUsage("database_host", pcode, quantity);
        }
        public static ProjectUsage forLog(long pcode, int quantity) {
            return new ProjectUsage("log", pcode, quantity);
        }
        public static ProjectUsage forDBCore(long pcode, int quantity) {
            return new ProjectUsage("database_vcpu", pcode, quantity);
        }
        public static ProjectUsage forBrowserSession(long pcode, int quantity) {
            return new ProjectUsage("browser_session", pcode, quantity);
        }
    
        private ProjectUsage(String identifier, long pcode, int quantity) {
            this.identifier = identifier;
            this.pcode = pcode;
            this.quantity = quantity;
            this.tags = new ArrayList<>();
            this.addTag("pcode", String.valueOf(this.pcode));
            this.addTag("identifier", identifier);
        }

        /**
         * Two UsageAllocations can't have the same tags (that is, the same combination of tag keys and values).
         * If that's the case, they must use the same UsageAllocation.
         */
        public UsageAllocation toUsageAllocation(){
            return new UsageAllocation()
                    .withAllocatedUsageQuantity(quantity)
                    .withTags(tags);
        }

        /**
         * Characters allowed for the tag key and value – a-zA-Z0-9+ -=._:\/@
         */
        public void addTag(String key, String value){
            this.tags.add(new Tag()
                    .withKey(key)
                    .withValue(value));
        }
    }
}
