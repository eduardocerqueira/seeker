//date: 2025-11-12T16:52:27Z
//url: https://api.github.com/gists/a36962462c5d61fed63a1ddbc2bd4cde
//owner: https://api.github.com/users/chriswang0427

package com.xgen.cloud.brs.web._private.util;

import com.xgen.cloud.brs.core._public.exception.BackupErrorCode;
import com.xgen.svc.core.model.api.SimpleApiResponse;
import com.xgen.svc.core.model.api.SimpleApiResponse.Builder;
import jakarta.ws.rs.core.Response;

public class WTCheckpointResponseBuilder {
  private static Builder getHttpResponseFromError(BackupErrorCode errorCode) {
    switch (errorCode) {
      case BACKUPS_NOT_ENABLED:
      case DUPLICATE_BACKUP_ID:
      case INSUFFICIENT_AGENT_VERSION:
      case INVALID_AGENT_VERSION:
      case INVALID_REQUEST:
      case MISMATCHED_BACKUP_ID:
      case MULTIPLE_CONCURRENT_SNAPSHOTS:
      case TARGETING_MISMATCH:
      case INVALID_MONGODB_VERSION:
        return SimpleApiResponse.badRequest(errorCode);
      case BACKUP_CURSOR_EXTEND_TIMEDOUT:
      case SNAPSHOT_SKIPPED:
        return SimpleApiResponse.failedDependency(errorCode);
      case INSUFFICIENT_FCV:
        return SimpleApiResponse.forbidden(errorCode);
      case BOUNDARY_DOES_NOT_EXIST:
      case PERSISTENCE_FAILURE:
      case CLUSTERSHOT_ABORTED:
      case SNAPSHOT_ABORTED:
      case BACKUP_NOT_ACTIVE:
      case WRONG_TAG:
        return SimpleApiResponse.non500Error(errorCode);
      case BACKUP_ID_NOT_FOUND:
      case JOB_NOT_FOUND:
      case CANT_RETRIEVE_RESTORE_INFO:
        return SimpleApiResponse.notFound(errorCode);
      case TOO_MANY_REQUESTS:
        return SimpleApiResponse.tooManyRequests(errorCode);
      case CANNOT_CHECK_FOR_ABANDONED_SNAPSHOTS:
      case CONCURRENT_GROOM_AND_SNAPSHOT:
      case RETRYABLE_BLOCK_HANDLING_ERROR:
      default:
        return SimpleApiResponse.error(errorCode);
    }
  }

  public static Response buildHttpResponseFromError(BackupErrorCode errorCode) {
    return getHttpResponseFromError(errorCode).build();
  }

  public static Response buildHttpResponseFromError(BackupErrorCode errorCode, String message) {
    return getHttpResponseFromError(errorCode).message(message).build();
  }

  public static Response buildHttpResponseFromErrorWithResource(
      BackupErrorCode errorCode, String resource) {
    return getHttpResponseFromError(errorCode).resource(resource).build();
  }

  public static Response buildOkHttpResponse() {
    return SimpleApiResponse.ok().build();
  }
}
