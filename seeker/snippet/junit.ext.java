//date: 2023-08-02T17:06:27Z
//url: https://api.github.com/gists/57da73346382cb6b495ac373225ed8f0
//owner: https://api.github.com/users/totalwar

public class TestExtension implements ParameterResolver, TestInstancePostProcessor, BeforeAllCallback {

    private static final ExtensionContext.Namespace EXT_NS = ExtensionContext.Namespace.create(PetriTestExtension.class);
    private static final String TEST_ENGINE = "TEST_ENGINE";

    @Override
    public boolean supportsParameter(ParameterContext paramCtx, ExtensionContext context) throws ParameterResolutionException {
        Parameter parameter = paramCtx.getParameter();
        if (parameter != null) {
            return parameter.getType().isAssignableFrom(TestEngine.class);
        }
        return false;
    }

    @Override
    public Object resolveParameter(ParameterContext paramCtx, ExtensionContext context) throws ParameterResolutionException {
        Parameter parameter = paramCtx.getParameter();
        if (parameter != null && parameter.getType().isAssignableFrom(TestEngine.class)) {
            return context.getStore(EXT_NS).get(TEST_ENGINE);
        }
        return null;
    }

    @Override
    public void postProcessTestInstance(Object testInstance, ExtensionContext context) throws Exception {
        DefaultTestEngine testEngine = context.getStore(EXT_NS).get(TEST_ENGINE, DefaultTestEngine.class);
        Method[] methods = testInstance.getClass().getDeclaredMethods();
        for (Method method : methods) {
            OnPetriTransition annotation = method.getAnnotation(OnTransition.class);
            if (annotation != null) {
                testEngine.getMethods().add(annotation.value());
            }
        }
    }

    @Override
    public void beforeAll(ExtensionContext context) throws Exception {
        DefaultTestEngine testEngine = new DefaultTestEngine();
        context.getStore(EXT_NS).put(TEST_ENGINE, testEngine);
    }
}