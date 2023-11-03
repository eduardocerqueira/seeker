#date: 2023-11-03T16:54:20Z
#url: https://api.github.com/gists/569080d228f00973c7ab0e8497d121a7
#owner: https://api.github.com/users/JulienDanh-Partoo

import libcst as cst
import libcst.matchers as m

m_view_config_decorated_function = m.FunctionDef(decorators=[m.Decorator(decorator=m.Call(func=m.Name("view_config")))])
m_request_current_user_assign = m.Assign(
    value=m.Attribute(
        value=m.Name("request"),
        attr=m.Name("current_user")
    )
)


class TransformUserDependency(m.MatcherDecoratableTransformer):
    """
    Transforme a module by:
    - Adding a UserDependency to the function params of a function decorated with @view_config
    - Removing the assignment to request.current_user

    For the purpose of example we simplify it to not handle:
    - Multiple assignations
    - Multiple decorators
    - Direct access to request.current_user
    - Import management
    """

    def __init__(self):
        super().__init__()
        self.user_variable: cst.Name | None = None

    @m.call_if_inside(m_view_config_decorated_function)
    def visit_FunctionDef(self, node: cst.FunctionDef) -> cst.FunctionDef:
        """
        On visiting a function decorated with @view_config we reset the user_variable
        """
        self.user_variable = None
        return node

    @m.call_if_inside(m_view_config_decorated_function)
    @m.leave(m_request_current_user_assign)
    def leave_AssignCurrentUser(self, original_node: cst.Assign,
                                updated_node: cst.Assign) -> cst.Assign | cst.RemovalSentinel:
        """
        On leaving an assignment to request.current_user we store the variable name and remove it from the tree
        """
        self.user_variable = original_node.targets[0].target
        return cst.RemoveFromParent()

    @m.call_if_inside(m_view_config_decorated_function)
    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        """
        On leaving a function decorated with @view_config we add the user variable to the function params
        """
        if self.user_variable:
            return updated_node.with_changes(
                params=updated_node.params.with_changes(
                    params=list(original_node.params.params) + [
                        cst.Param(self.user_variable, cst.Annotation(cst.Name("UserDependency")))]
                )
            )

        return updated_node


def test_transformer():
    input = """
@view_config(route_name="some_route", renderer="json", required_role="ADMIN")
def my_controller(request: Request):
    user = request.current_user
    # Some code using request

    return {"key": "value"}
"""

    output_module = cst.parse_module(input).visit(TransformUserDependency())

    assert output_module.code == """
@view_config(route_name="some_route", renderer="json", required_role="ADMIN")
def my_controller(request: Request, user: UserDependency):
    # Some code using request

    return {"key": "value"}
"""
