//date: 2024-01-31T17:06:13Z
//url: https://api.github.com/gists/9d08f1ae072d55cc8b4361afe7177631
//owner: https://api.github.com/users/alex-jurgens

package com.constructsecure.bahadur.newoutreach.service;

import java.net.MalformedURLException;
import java.util.List;
import java.util.Set;

import com.querydsl.core.BooleanBuilder;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;

import static com.constructsecure.bahadur.Utility.getClient;
import static com.constructsecure.bahadur.Utility.getUser;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.when;

import com.constructsecure.bahadur.Utility;
import com.constructsecure.bahadur.authentication.service.UserService;
import com.constructsecure.bahadur.client.services.ClientService;
import com.constructsecure.bahadur.common.factory.FactoryBeanResolver;
import com.constructsecure.bahadur.common.model.KeyValueDTO;
import com.constructsecure.bahadur.newoutreach.model.MasterOpportunityFilterResponse;
import com.constructsecure.bahadur.newoutreach.model.MasterOpportunitySearchRequest;
import com.constructsecure.bahadur.newoutreach.model.MasterOpportunitySource;
import com.constructsecure.bahadur.newoutreach.model.MasterOpportunityType;
import com.constructsecure.bahadur.util.CommonUtil;
import com.highwire.common.api.model.request.PageableRequestParams;
import org.springframework.data.domain.Sort;

public class MasterOpportunityFilterServiceTest {

    @Mock
    private ClientService clientService;
    @Mock
    private UserService userService;
    @Mock
    private FactoryBeanResolver factoryBeanResolver;
    @InjectMocks
    private MasterOpportunityFilterService masterOpportunityFilterService;

    @BeforeEach
    void setUp() throws MalformedURLException {
        Utility.initMocksWithFactoryBeanResolver(this);
    }

    @Test
    void masterOpportunityFilterServiceTest() {
        // set up owner data
        KeyValueDTO userKeyValueDTO = new KeyValueDTO();
        userKeyValueDTO.setKey(getUser().getUserId());
        userKeyValueDTO.setValue(CommonUtil.getFullName(getUser().getFirstName(),
                getUser().getLastName()));
        when(userService.getCpeUsersKeyValueFormat()).thenReturn(List.of(userKeyValueDTO));
        // set up client data
        KeyValueDTO clientKeyValueDTO = new KeyValueDTO();
        clientKeyValueDTO.setKey(getClient().getClientId());
        clientKeyValueDTO.setValue(getClient().getCompanyName());
        when(clientService.getAllClientsKeyValue()).thenReturn(List.of(clientKeyValueDTO));

        // call the service
        MasterOpportunityFilterResponse filterResponse = masterOpportunityFilterService.getFilters();
        assertNotNull(filterResponse);
        assertEquals(filterResponse.owners().size(), 1);
        assertEquals(filterResponse.hiringPartners().size(), 1);
        assertEquals(filterResponse.opportunityTypes().size(), MasterOpportunityType.values().length);
        assertEquals(filterResponse.opportunitySources().size(), MasterOpportunitySource.values().length);
    }

    @Test
    void testApplyUserFiltersForSearchText() {
        PageableRequestParams params = new PageableRequestParams(0, 10, null);
        String searchText = "searchString";
        String expectedResult = "contains(masterOpportunity.contractor.companyName," + searchText + ")";
        MasterOpportunitySearchRequest searchRequest = new MasterOpportunitySearchRequest(params, Set.of(), Set.of(),
                Set.of(), Set.of(), Set.of(), searchText, "", "");
        BooleanBuilder booleanBuilder = new BooleanBuilder();
        masterOpportunityFilterService.applyUserFilters(booleanBuilder, searchRequest, null);
        assert booleanBuilder.getValue() != null;
        assertTrue(booleanBuilder.getValue().toString().contains(expectedResult));
    }

    @Test
    void testApplyUserSortOrder() {
        PageableRequestParams params = new PageableRequestParams(0, 10, null);
        MasterOpportunitySearchRequest searchRequest = new MasterOpportunitySearchRequest(params, Set.of(), Set.of(),
                Set.of(), Set.of(), Set.of(), "", "createdDate", "asc");
        Sort.Order result = masterOpportunityFilterService.getUserSortOrder(searchRequest);
        assertTrue(result.equals(Sort.Order.asc("createdDate")));
    }
}
