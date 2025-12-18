# System Design Document Template

This document is a template for a system design document. It is meant to be used as a starting point.

## Context and Scope

This section gives the reader a very rough overview of the landscape in which the new system is being built and what is being built.

### Goals and non-goals

A short list of bullet points of what the goals of the system are, and, sometimes more importantly, what non-goals are. Note, that non-goals aren’t negated goals like “The system shouldn’t crash”, but rather things that could reasonably be goals, but are explicitly chosen not to be goals.

### Scenarios

In designing solutions, it helps to imagine a few real-life stories of how actual (stereotypical) people would use them.

## System Overview

This section should start with an overview and then go into details.

### Assumptions and Dependencies

Description of any assumptions that may be wrong or any dependencies.

### General Constraints

Enumerate all the things you can do relatively easily, but you need to creatively put those things together to achieve the goals. There may be multiple solutions, and none of them are great, and hence such a document should focus on selecting the best way given all identified trade-offs.

### APIs

Describe APIs abstracting the business logic of this system.

### Data storage

Discuss how and in what rough form data is going to be stored.

### Code and Contracts

Describe novel algorithms and API contracts for system integration.

### Alternatives considered

This section lists alternative designs that would have reasonably achieved similar outcomes. The focus should be on the trade-offs that each respective design makes and how those trade-offs led to the decision to select the design that is the primary topic of the document.

## Operational Excellence

Ensure that certain cross-cutting concerns such as security, privacy, and observability are always taken into consideration.

### Testing

Describe unit, integration, end to end and regression testing strategy.

### Release

Describe how features are released such as regions, feature flags, user base.

### Observability

Describe how logs, alerts and KPIs are going to be setup.

## Security & Privacy

Describe security of the system, user, and data privacy

## User Interface

Append high level UI/UX wireframes.

## Cost Analysis

Describe total cost of ownership and infrastructure cost estimates.
