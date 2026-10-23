# Multi-Agent and Agentic AI Systems

An agentic AI system is one where a language model does not just answer a
single prompt, but plans, calls tools, observes results, and decides what
to do next -- often across multiple steps. A multi-agent system takes this
further by splitting responsibilities across several specialised agents
that collaborate, each with a narrower role, rather than asking one large
prompt to do everything at once.

A common pattern is the router-worker-critic pattern. A router agent first
classifies the incoming request and decides which path to take. One or more
worker agents then perform the actual task -- for example, a retriever agent
that searches a knowledge base, or a synthesizer agent that drafts an
answer. Finally, a critic agent evaluates the worker's output against a
set of quality checks; if the output fails, the critic can send it back for
another attempt with feedback, instead of returning a low-quality result to
the user. This retry loop is bounded by a maximum number of attempts to
avoid infinite loops and runaway cost.

Frameworks such as LangGraph model this kind of workflow explicitly as a
graph: each agent is a node, and edges (including conditional edges) define
how control passes between agents based on the current state. This makes
the control flow of a multi-agent system explicit, testable, and easy to
extend with new nodes -- for example, adding a new guardrail agent without
having to rewrite the whole pipeline.

Good multi-agent systems keep individual agents narrow and composable,
make the state passed between agents explicit, and always define a clear
termination condition so that a disagreement between agents cannot loop
forever.
