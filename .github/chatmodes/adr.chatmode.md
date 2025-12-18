# Create Architectural Decision Record

Create an ADR document for `${input:DecisionTitle}` using structured formatting optimized for AI consumption and human readability.

## Inputs

- **Decision Title**: `${input:DecisionTitle}` - Brief, descriptive title for the architectural decision
- **Stakeholders**: `${input:Stakeholders}` - People, teams, or roles involved in or affected by this decision
- **Context**: `${input:Context}` - Problem statement, technical constraints, business requirements, and environmental factors
- **Decision Drivers**: `${input:DecisionDrivers}` - Key factors influencing the decision (performance, cost, maintainability, etc.)
- **Alternatives**: `${input:Alternatives}` - Other options considered with how they address decision drivers and rejection rationale
- **Chosen Decision**: `${input:ChosenDecision}` - The selected solution and rationale
- **Consequences**: `${input:Consequences}` - Technical summary of outcomes and trade-offs organized by decision driver categories
- **Related Decisions** (optional): `${input:RelatedDecisions}` - ADRs that this supersedes or relates to

## Input Validation

If any of the required inputs are not provided or cannot be determined from the conversation history, ask the user to provide the missing information before proceeding with ADR generation.

## Requirements

- Use precise, unambiguous language
- Follow standardized ADR format with front matter
- Present consequences as a balanced technical summary organized by decision drivers
- Document alternatives with how they address decision drivers and rejection rationale
- Structure for machine parsing and human reference
- Organize content by categorical decision drivers (Performance, Cost, Technical Impact, Team & Operations, Compliance & Security)
- Show clear relationships between drivers, alternatives, and consequences
- Link to related ADRs and external references

The ADR must be saved in the `/docs/adr/` directory using the naming convention: `adr-NNNN-[title-slug].md`, where NNNN is the next sequential 4-digit number (e.g., `adr-0001-database-selection.md`).

## Numbering Guidelines

1. Check existing ADRs in `/docs/adr/` to determine the next sequential number
2. Use 4-digit zero-padded format (e.g., 0001, 0002, 0023, 0100)
3. Create a URL-friendly slug from the decision title (lowercase, hyphen-separated)
4. Example: "Use PostgreSQL for Data Storage" → `adr-0012-use-postgresql-for-data-storage.md`

## Writing Guidelines

1. Use bold text with colons instead of markdown headers for easier review.
1. Prefer clean, concise summaries over verbose, repetitive explanations.
1. Use tables where necessary for comparing the numbers and benchmarks

## Required Documentation Structure

The documentation file must follow the template below, ensuring that all sections are filled out appropriately. The front matter for the markdown should be structured correctly as per the example following:

```md
---
title: "ADR-NNNN: [Decision Title]"
status: "Proposed"
date: "YYYY-MM-DD"
authors: "[Stakeholder Names/Roles]"
tags: ["architecture", "decision"]
supersedes: ""
superseded_by: ""
---

# ADR-NNNN: [Decision Title]

## Status

**Proposed** | **Accepted** | **Rejected** | **Superseded** | **Deprecated**

_Status Guide:_
- **Proposed**: Under review, not yet approved
- **Accepted**: Approved and ready for implementation
- **Rejected**: Considered but not approved
- **Superseded**: Replaced by a newer ADR (reference in superseded_by)
- **Deprecated**: No longer recommended but not formally replaced

## Context

[Provide a comprehensive problem statement that explains the situation requiring a decision. Include:]
- Technical constraints and limitations
- Business requirements and objectives
- Current system state and pain points
- Environmental factors (team skills, timeline, budget, regulations)
- Stakeholder concerns and priorities

## Decision Drivers

### Performance Requirements
[e.g., "Need for sub-100ms response times", "Must handle 10,000 concurrent users"]

### Cost Constraints
[e.g., "Budget limitation of $X per month", "Total cost of ownership over 3 years"]

### Technical Constraints
[e.g., "Must integrate with existing Python/Django stack", "Limited to cloud-native solutions"]

### Team Capabilities
[e.g., "Team has expertise in Python/JavaScript", "Limited DevOps resources available"]

### Compliance & Security
[e.g., "Must meet GDPR requirements", "SOC 2 Type II certification required"]

### Scalability & Maintainability
[e.g., "Must scale to 10x current load", "Need to reduce maintenance overhead"]

## Decision

[State the chosen solution clearly and concisely. Explain:]
- What solution was selected
- Why this solution best addresses the decision drivers
- How it solves the problem described in the context
- Key technical approach or architecture
- Any critical assumptions or dependencies

## Consequences

This section describes the technical outcomes and trade-offs of implementing this decision, organized by the decision driver categories. Each category should include both benefits and limitations to provide a balanced view.

### Performance

[Describe how this decision impacts performance requirements. Include:]
- Expected performance characteristics (latency, throughput, resource usage)
- Performance improvements achieved compared to current state
- Performance limitations or bottlenecks introduced
- Scenarios where performance may degrade
- Scalability implications

### Cost

[Describe how this decision impacts cost constraints. Include:]
- Initial implementation costs (development time, infrastructure setup)
- Ongoing operational costs (hosting, maintenance, licensing)
- Cost savings or optimizations compared to alternatives
- Hidden or future costs to consider
- Cost-benefit analysis summary

### Technical Impact

[Describe how this decision affects the technical landscape. Include:]
- Integration with existing systems and architecture
- Technical complexity introduced or reduced
- Dependencies on specific technologies, frameworks, or vendors
- Technical debt created or eliminated
- Maintenance and operational considerations
- Testing and deployment implications

### Team & Operations

[Describe how this decision affects the team and operations. Include:]
- Required skills and expertise (existing vs. need to acquire)
- Learning curve and training requirements
- Impact on developer experience and productivity
- Changes to workflows, processes, or tooling
- Support and on-call burden
- Documentation and knowledge sharing needs

### Compliance & Security

[Describe how this decision impacts compliance and security. Include:]
- Security posture improvements or new vulnerabilities
- Compliance requirements met or gaps remaining
- Data privacy and protection implications
- Audit trail and monitoring capabilities
- Security maintenance and patching requirements

## Alternatives Considered

### [Alternative 1 Name]

#### Description
[Brief technical description of the alternative solution]

#### How It Addresses Decision Drivers

##### Performance
[How this alternative would address performance requirements, or why it falls short]

##### Cost
[How this alternative would address cost constraints, or why it falls short]

##### Technical Fit
[How this alternative would address technical constraints, or why it falls short]

##### Team & Operations
[How this alternative would leverage team capabilities, or why it falls short]

##### Compliance & Security
[How this alternative would address compliance/security requirements, or why it falls short]

#### Rejection Rationale
[Primary reasons why this option was not selected, referencing specific decision drivers]

### [Alternative 2 Name]

#### Description
[Brief technical description of the alternative solution]

#### How It Addresses Decision Drivers

##### Performance
[How this alternative would address performance requirements, or why it falls short]

##### Cost
[How this alternative would address cost constraints, or why it falls short]

##### Technical Fit
[How this alternative would address technical constraints, or why it falls short]

##### Team & Operations
[How this alternative would leverage team capabilities, or why it falls short]

##### Compliance & Security
[How this alternative would address compliance/security requirements, or why it falls short]

#### Rejection Rationale
[Primary reasons why this option was not selected, referencing specific decision drivers]

## References

- **REF-001**: [Related ADRs, e.g., "ADR-0005: API Gateway Selection"]
- **REF-002**: [External documentation, standards, or specifications]
- **REF-003**: [Frameworks, libraries, or tools referenced]
- **REF-004**: [Research papers, blog posts, or case studies]
- **REF-005**: [Vendor documentation or official guidelines]

## Review and Approval

Before finalizing the ADR:
1. Ensure all stakeholders have reviewed and provided input
2. Verify that all decision drivers are addressed in the decision, consequences, and alternatives
3. Confirm that consequences present a balanced, realistic technical summary with both benefits and trade-offs
4. Validate that alternatives show clear comparison against decision drivers
5. Check that rejection rationale for alternatives references specific drivers
```

## Writing Guidelines

### For Context Section

- Be specific about the problem being solved
- Quantify constraints when possible (e.g., "Handle 10,000 requests/second")
- Explain the business impact or urgency
- Describe what happens if no decision is made

### For Decision Drivers

- Organize drivers into categorical sections (Performance, Cost, Technical, Team, Compliance, Scalability)
- Be specific and measurable when possible (e.g., "< 100ms response time" not "fast")
- Include both technical and business considerations
- Only include categories that are relevant to the decision
- Prioritize the most critical drivers within each category

### For Alternatives

- Document at least 2-3 alternatives seriously considered
- Organize how each alternative addresses decision drivers using the same categories
- Explain rejection rationale by referencing specific decision drivers
- Include "Do Nothing" as an alternative if relevant
- Only include driver categories that are relevant to each alternative
- Reference real-world examples or case studies when available
- Show why the chosen decision better addresses the drivers than alternatives

### For Consequences

- Organize consequences by decision driver categories (Performance, Cost, Technical Impact, Team & Operations, Compliance & Security)
- Present a balanced technical summary including both benefits and trade-offs for each category
- Avoid explicit "positive" vs "negative" labeling - describe the reality of what this decision brings
- Be specific and quantify impacts when possible (e.g., "reduces latency by ~30ms" or "adds 2 weeks to initial development")
- Include both immediate and long-term implications
- Describe operational impacts (monitoring, maintenance, support burden)
- Consider second-order effects (e.g., "faster deployment may increase incident rate initially")
- Only include categories that are relevant to the decision
- Show clear relationships between decision drivers and outcomes



## Common Tags

Use appropriate tags in the front matter to categorize ADRs:

- `["architecture", "decision"]` - Default for all ADRs
- `["infrastructure", "cloud"]` - Infrastructure decisions
- `["database", "storage"]` - Data storage decisions
- `["security", "compliance"]` - Security-related decisions
- `["api", "integration"]` - API and integration decisions
- `["frontend", "ui"]` - Frontend architecture decisions
- `["backend", "services"]` - Backend service decisions
- `["observability", "monitoring"]` - Monitoring and logging decisions
- `["performance", "scalability"]` - Performance-related decisions
- `["tooling", "developer-experience"]` - Development tools and processes

## Post-Creation Tasks

After creating an ADR:

1. Commit the ADR to version control
2. Update any related documentation (README, architecture diagrams)
3. Communicate the decision to affected teams
4. Schedule a review meeting if the decision is controversial
5. Add the ADR to the team's knowledge base or wiki
6. Update project roadmap or sprint planning based on implementation notes
