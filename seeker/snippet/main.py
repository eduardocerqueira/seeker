#date: 2025-11-14T16:51:15Z
#url: https://api.github.com/gists/e27c3bee8c72bb16bbce11048b659672
#owner: https://api.github.com/users/ryblogs

import panel as pn
import param

pn.extension()


class Dashboard(param.Parameterized):
    """Minimal reactive dashboard with status sidebar."""

    # Parameters (these automatically become widgets)
    name = param.String(default="World", doc="Enter your name")
    count = param.Integer(default=0, bounds=(0, 100), doc="Select a count")
    color = param.Selector(default="blue", objects=["red", "blue", "green"])

    # Status tracking
    last_action = param.String(default="Initialized")
    total_updates = param.Integer(default=0)

    @param.depends("name", "count", "color", watch=True)
    def _update_status(self):
        """Reactive callback that fires when any parameter changes."""
        self.total_updates += 1
        self.last_action = f"name={self.name}, count={self.count}, color={self.color}"

    @param.depends("name", "count")
    def main_view(self):
        """Main content area - updates reactively."""
        return pn.pane.Markdown(f"""
        ## Hello, {self.name}! 🎉

        Your count is: **{self.count}**

        Current color: **{self.color}**

        ---
        *This updates automatically when parameters change!*
        """)

    @param.depends("last_action", "total_updates")
    def status_sidebar(self):
        """Status sidebar - updates reactively."""
        last_action = self.last_action
        total_updates = self.total_updates
        return pn.Column(
            pn.pane.Markdown("### 📊 Status"),
            pn.pane.Markdown(f"**Updates:** {total_updates}"),
            pn.pane.Markdown(f"**Last Action:**"),
            pn.pane.Markdown(
                f"```\n{last_action[:80]}...\n```"
                if len(last_action) > 80
                else f"```\n{last_action}\n```"
            ),
            styles={"background": "#f0f0f0", "padding": "10px", "border-radius": "5px"},
        )

    def view(self):
        """Assemble the complete dashboard."""
        return pn.template.FastListTemplate(
            title="Reactive Dashboard MVP",
            sidebar=[
                pn.pane.Markdown("### ⚙️ Controls"),
                self.param,  # Auto-generates widgets for all params
                pn.layout.Divider(),
                self.status_sidebar,  # Reactive status display
            ],
            main=[
                self.main_view,  # Reactive main content
            ],
            accent="#3b82f6",
        )


# Create and serve
dashboard = Dashboard()
dashboard.view().servable()
