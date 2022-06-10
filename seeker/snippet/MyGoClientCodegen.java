//date: 2022-06-10T17:13:21Z
//url: https://api.github.com/gists/6b84046488c4d62ac0eb4110a9cb52e8
//owner: https://api.github.com/users/gcatanese

package org.mycompany.codegen;

import org.openapitools.codegen.CodegenOperation;
import org.openapitools.codegen.CodegenParameter;
import org.openapitools.codegen.model.ModelMap;
import org.openapitools.codegen.model.OperationMap;
import org.openapitools.codegen.model.OperationsMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class MyGoClientCodegen extends GoClientCodegen {

    private final Logger LOGGER = LoggerFactory.getLogger(MyGoClientCodegen.class);


    @Override
    public OperationsMap postProcessOperationsWithModels(OperationsMap objs, List<ModelMap> allModels) {
        OperationsMap operationsMap = super.postProcessOperationsWithModels(objs, allModels);

        OperationMap operations = objs.getOperations();

        List<CodegenOperation> operationList = operations.getOperation();
        for (CodegenOperation op : operationList) {
            LOGGER.info(op.operationId);

            // renaming operationId (remove verb prefix)
            op.operationId = op.operationId.replace("Post", "");
            op.operationId = op.operationId.replace("Get", "");
            op.operationId = op.operationId.replace("Patch", "");
            op.operationId = op.operationId.replace("Put", "");

            op.operationId = op.operationId.substring(0, 1).toLowerCase() + op.operationId.substring(1);

        }

        return operationsMap;
    }

}