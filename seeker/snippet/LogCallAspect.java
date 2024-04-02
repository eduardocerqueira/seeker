//date: 2024-04-02T16:52:29Z
//url: https://api.github.com/gists/d7b5d5c3109c645fa298ada0ff30aba6
//owner: https://api.github.com/users/leafriend

import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.Signature;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.reflect.MethodSignature;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.aop.aspectj.MethodInvocationProceedingJoinPoint;
import org.springframework.stereotype.Component;

@Aspect
@Component
public class LogCallAspect {

	@Around("@annotation(com.example.springbootpg.web.LogCall)")
	public Object logCall(ProceedingJoinPoint joinPoint) throws Throwable {
		Object[] args = joinPoint.getArgs();
		if (joinPoint instanceof MethodInvocationProceedingJoinPoint m) {
			Object target = m.getTarget();
			Logger logger = LoggerFactory.getLogger(target.getClass());
			Signature signature = m.getSignature();
			if (signature  instanceof MethodSignature ms) {
				ms.getMethod();
				String methodName = ms.getName();
				String[] parameterNames = ms.getParameterNames();
				for (int i = 0; i < parameterNames.length; i++) {
					logger .debug("{}() <<< {} = {}", methodName, parameterNames[i], args[i]);
				}

				try {

					Object result = joinPoint.proceed();
					logger .debug("{}() >>> {}", methodName, result);
					return result;
				} catch (final Throwable t) {
					logger .debug("{}() !!! {}", methodName, t.getMessage(), t);

				}
			}
		}
		// Signature signature = joinPoint.getSignature();
		// signature.
		return joinPoint.proceed();
	}
}

