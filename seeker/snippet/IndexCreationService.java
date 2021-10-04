//date: 2021-10-04T16:55:09Z
//url: https://api.github.com/gists/d446ddf74e4e01bd231b319cbd76cdb9
//owner: https://api.github.com/users/senoritadeveloper01

package com.senoritadev.archive.processor.service.index;

import com.senoritadev.archive.processor.exception.ArchiveException;
import com.senoritadev.archive.processor.model.ArchiveErrorCode;
import com.senoritadev.archive.processor.model.constant.IndexConstants;
import com.senoritadev.archive.processor.util.CommonUtils;
import com.senoritadev.archive.processor.util.JsonTemplateUtil;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.elasticsearch.client.RequestOptions;
import org.elasticsearch.client.RestHighLevelClient;
import org.elasticsearch.client.indices.CreateIndexRequest;
import org.elasticsearch.client.indices.CreateIndexResponse;
import org.elasticsearch.common.settings.Settings;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
public class IndexCreationService {

    @Value("${elasticsearch.index.number_of_shards:1}")
    private Integer numberOfShards;

    @Value("${elasticsearch.index.number_of_replicas:0}")
    private Integer numberOfReplicas;

    private final RestHighLevelClient client;

    private final JsonTemplateUtil jsonTemplateUtil;


    /**
     * Creates session index using the special mapping.
     *
     * @return
     */
    public boolean createOrganizationIndex() {
        try {
            Map<String, Object> mappingSource = jsonTemplateUtil
                    .readJsonTemplateAsMap(toJsonTemplatePath(IndexConstants.MAIL_ITEM_INDEX_MAPPING));

            Map<String, Object> settingSource = jsonTemplateUtil
                    .readJsonTemplateAsMap(toJsonTemplatePath(IndexConstants.MAIL_ITEM_INDEX_SETTING));

            CreateIndexRequest request = new CreateIndexRequest(IndexConstants.INDEX_NAME);

            request.settings(Settings.builder()
                    .put("index.number_of_shards", numberOfShards)
                    .put("index.number_of_replicas", numberOfReplicas)
            );

            request.mapping(mappingSource);
            request.settings(settingSource);

            CreateIndexResponse createIndexResponse = client.indices().create(request, RequestOptions.DEFAULT);

            return createIndexResponse.isAcknowledged();

        } catch (Exception e) {
            ArchiveException ae = new ArchiveException(ArchiveErrorCode.ELASTICSEARCH_ACTION_FAILED, CommonUtils.buildStackTraceString(e));
            log.error("[ELASTICSEARCH_SERVICE] [CREATE_INDEX] [EXCEPTION] [CAUSE: {}]", CommonUtils.buildStackTraceString(ae));
            throw ae;
        }
    }

    /**
     * Generates JSON template path for the given template filename.
     *
     * @param templateFilename
     * @return
     */
    private String toJsonTemplatePath(String templateFilename) {
        StringBuilder sb = new StringBuilder();
        sb.append(IndexConstants.ELASTICSEARCH_JSON_TEMPLATE_ROOT_PATH);
        sb.append(templateFilename);

        return sb.toString();
    }
}