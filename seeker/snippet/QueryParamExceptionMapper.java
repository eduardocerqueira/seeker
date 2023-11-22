//date: 2023-11-22T16:41:17Z
//url: https://api.github.com/gists/29ca50f419e5ba23fa4f6f827ed03653
//owner: https://api.github.com/users/BasiaRejdychIwanek

package com.imggaming.dce.commons.framework.dropwizard.errorhandling.exceptionmappers.recoverable;

import com.imggaming.dce.commons.framework.dropwizard.IMGErrorMessageFactory;
import com.imggaming.dce.commons.framework.dropwizard.config.RecoverableErrorStackTrace;
import com.imggaming.dce.commons.framework.dropwizard.errorhandling.exceptionmappers.IMGExceptionMapper;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.ws.rs.core.Response;
import org.glassfish.jersey.server.ParamException;

/** Created by gluiz on 15/03/2017. */
public class QueryParamExceptionMapper
    extends IMGExceptionMapper<ParamException.QueryParamException> {
  private final String[] acceptableValues;

  public QueryParamExceptionMapper(
      IMGErrorMessageFactory factory, RecoverableErrorStackTrace recoverableErrorStackTrace) {
    this(factory, new String[] {}, recoverableErrorStackTrace);
  }

  public QueryParamExceptionMapper(
      IMGErrorMessageFactory factory,
      String[] acceptableValues,
      RecoverableErrorStackTrace recoverableErrorStackTrace) {
    super(factory, recoverableErrorStackTrace);
    this.acceptableValues = acceptableValues == null ? new String[0] : acceptableValues;
  }

  @Override
  public String[] messages(ParamException.QueryParamException e) {
    return new String[] {
      String.format(
          "%s is invalid. Acceptable values are: %s",
          e.getParameterName(),
          acceptableValues.length > 1
              ? Stream.of(this.acceptableValues).collect(Collectors.joining(",", "[", "]"))
              : "not specified")
    };
  }

  @Override
  public int status(ParamException.QueryParamException e) {
    return Response.Status.BAD_REQUEST.getStatusCode();
  }
}
