//date: 2022-05-03T17:19:02Z
//url: https://api.github.com/gists/ba8867a53f98ea016ea3dffbf13c2d64
//owner: https://api.github.com/users/kirshiyin89

@ArchTag("example")
@AnalyzeClasses(packages = "com.tngtech.archunit.example.layers")
public class ControllerRulesTest {

    @ArchTest
    static final ArchRule controllers_should_only_call_secured_methods =
            classes().that().resideInAPackage("..controller..")
                    .should().onlyCallMethodsThat(areDeclaredInController().or(are(annotatedWith(Secured.class))));

    @ArchTest
    static final ArchRule controllers_should_only_call_secured_constructors =
            classes()
                    .that().resideInAPackage("..controller..")
                    .should().onlyCallConstructorsThat(areDeclaredInController().or(are(annotatedWith(Secured.class))));

