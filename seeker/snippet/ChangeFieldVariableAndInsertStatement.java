//date: 2022-03-10T16:58:03Z
//url: https://api.github.com/gists/88d6141220c76390ccc030bb1b3ab1ae
//owner: https://api.github.com/users/pway99

package org.openrewrite.java;

import org.openrewrite.ExecutionContext;
import org.openrewrite.Recipe;
import org.openrewrite.internal.lang.Nullable;
import org.openrewrite.java.tree.J;
import org.openrewrite.java.tree.TypeUtils;

public class MyRecipe extends Recipe {
    @Override
    public String getDisplayName() {
        return "ChangeAutowiredBooleanIsColorToColor";
    }

    @Override
    protected JavaIsoVisitor<ExecutionContext> getVisitor() {
        return new JavaIsoVisitor<ExecutionContext>() {
            final AnnotationMatcher autowiredAnnotationMatcher = new AnnotationMatcher("org.springframework.beans.factory.annotation.Autowired");

            @Override
            public J.VariableDeclarations.NamedVariable visitVariable(J.VariableDeclarations.NamedVariable variable, ExecutionContext executionContext) {
                J.VariableDeclarations.NamedVariable nv = super.visitVariable(variable, executionContext);
                if (hasAutowiredAnno(getCursor().firstEnclosing(J.VariableDeclarations.class))) {
                    if (TypeUtils.isOfClassType(nv.getType(), "java.lang.Boolean") && "isBlue".equals(nv.getSimpleName())) {
                        nv = nv.withName(nv.getName().withSimpleName("color"));
                    }
                }
                return nv;
            }

            @Override
            public J.VariableDeclarations visitVariableDeclarations(J.VariableDeclarations multiVariable, ExecutionContext ctx) {
                J.VariableDeclarations varDecls = super.visitVariableDeclarations(multiVariable, ctx);
                if (TypeUtils.isOfClassType(varDecls.getType(), "java.lang.Boolean")
                        && varDecls.getVariables().get(0).getSimpleName().equals("color")
                        && hasAutowiredAnno(varDecls)) {
                    varDecls = (J.VariableDeclarations)
                            new ChangeType("java.lang.Boolean", "java.awt.Color", true)
                                    .getVisitor().visitNonNull(varDecls, ctx);
                    maybeAddImport("java.awt.Color");
                    getCursor().dropParentUntil(J.Block.class::isInstance).putMessage("ADD_STATEMENT_AFTER", varDecls);
                }
                return varDecls;
            }

            @Override
            public J.Block visitBlock(J.Block block, ExecutionContext executionContext) {
                J.Block bl = super.visitBlock(block, executionContext);
                J.VariableDeclarations varDecls = getCursor().pollMessage("ADD_STATEMENT_AFTER");
                if (varDecls != null) {
                    JavaTemplate t = JavaTemplate.builder(this::getCursor, "Boolean isBlue = (#{any(java.awt.Color)} == Color.BLUE);")
                        .imports("java.awt.Color").build();
                    bl = bl.withTemplate(t, varDecls.getCoordinates().after(), varDecls.getVariables().get(0).getName());
                }
                return bl;
            }

            private boolean hasAutowiredAnno(@Nullable J.VariableDeclarations varDecls) {
                return varDecls != null && varDecls.getLeadingAnnotations().stream().anyMatch(autowiredAnnotationMatcher::matches);
            }
        };
    }
}
