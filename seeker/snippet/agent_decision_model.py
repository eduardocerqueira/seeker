#date: 2026-02-02T17:28:33Z
#url: https://api.github.com/gists/ebfdec94d1cb3304ef542b7a8e12c8cb
#owner: https://api.github.com/users/inURwhey

"""
PATTERN: Structured Agentic Memory (ADR).
Replaces legacy Markdown logs with a queryable database layer for autonomous agents.
"""

class AgentDecision(db.Model):
    """
    Stores strategic pivots made by agents (e.g., 'Penny' or 'Sentinel').
    Allows other agents to 'read the history' of the system.
    """
    id = db.Column(db.Integer, primary_key=True)
    agent_key = db.Column(db.String(50), db.ForeignKey('agent_profile.key'))
    title = db.Column(db.String(255))
    content = db.Column(db.Text)     # The 'What'
    rationale = db.Column(db.Text)   # The 'Why'
    impact = db.Column(db.Text)      # The 'Result'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_context_string(self):
        """Converts the decision into a format the LLM can ingest during a session."""
        return f"[{self.created_at.date()}] {self.title}: {self.content} (Rationale: {self.rationale})"