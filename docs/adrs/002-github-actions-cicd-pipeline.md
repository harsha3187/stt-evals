# ADR-0006: GitHub Actions CI/CD Pipeline with Service Principal Authentication and Multi-Environment Support

```md
status: "Proposed"
date: "2025-11-11"
authors: "ISE Oryx Crew"
tags: ["architecture", "decision", "cicd", "infrastructure", "security"]
supersedes: ""
superseded_by: ""
```

## Context

The project requires automated infrastructure deployment to Azure with support for multiple environments (production and integration testing). Current deployment is manual via bash scripts (`setup-remote-state.sh`, `provision.sh`, `generate-env.sh`), which lacks reproducibility, approval gates, and automated testing.

**Requirements:**

- Deploy Terraform infrastructure to dedicated Azure subscription (fresh subscription)
- CI/CD-only setup (no local developer Azure access required)
- Support production and integration test environments
- Enable on-demand integration environment creation for testing
- Secure authentication compatible with all GitHub Enterprise versions (including on-premises)
- Automated API integration testing post-deployment
- Audit trail and approval gates for production changes
- Setup must work on completely fresh Azure subscriptions
- Must work in customer environments without OIDC support

## Decision Drivers

### Security & Compliance

- Minimize secret sprawl and credential rotation burden
- Enforce approval gates for production deployments
- Complete audit trail for infrastructure changes

### Operational Efficiency

- Reduce manual deployment errors
- Enable faster testing cycles with on-demand environments
- Automated validation of infrastructure changes

### Technical Fit

- Existing .http files for manual API testing (HTTPYac)
- Terraform with Azure backend for state management
- GitHub as source control platform

### Team Capabilities

- Team familiar with GitHub Actions
- Existing bash automation scripts to integrate
- Limited DevOps resources (need low-maintenance solution)

## Decision

Implement **GitHub Actions CI/CD pipeline with Service Principal authentication and GitHub Environments** for multi-environment deployment on fresh Azure subscriptions.

**Architecture:**

1. **GitHub Environments**: `production` (main branch, approval required) and `integration` (all branches, on-demand)
2. **Service Principal Authentication**: Standard Azure AD service principals with client secrets, separate principals per environment (`oryx-cap-mve-prd-github-actions`, `oryx-cap-mve-int-github-actions`)
3. **Secret Management**: Client secrets stored as GitHub environment secrets with 90-day rotation policy
4. **HTTPYac CLI Testing**: Automated .http file execution with JUnit/JSON reporting
5. **Three Workflows**:
   - `terraform-plan.yml`: PR validation with plan comments
   - `terraform-deploy.yml`: Infrastructure deployment (prod/int)
   - `integration-tests.yml`: HTTPYac API tests post-deployment

**Key Features:**

- Works on fresh Azure subscriptions (no existing resources required)
- CI/CD-only setup (no individual developer Azure access)
- Compatible with all GitHub Enterprise versions (including on-premises, no OIDC required)
- Separate Terraform state files (`production.tfstate`, `integration.tfstate`)
- Automated SP setup via `setup-github-sp.sh` (idempotent, fresh subscription compatible)
- Automated GitHub configuration via `configure-github.sh`
- Environment-specific secrets and variables
- Artifact retention for Terraform outputs (30 days)
- Project naming: `oryx-cap-mve-prd` / `oryx-cap-mve-int` (Azure naming constraints)
- Client secret rotation guidance and automation support

## Consequences

### Security

**Benefits:** Standard Azure AD authentication pattern, scope-limited credentials, complete Azure AD audit logs, CI/CD-only access (no individual developer permissions needed), works in all GitHub Enterprise environments  
**Trade-offs:** Client secrets stored in GitHub (encrypted at rest), requires secret rotation policy (90-180 days recommended), requires Owner/User Access Administrator for initial setup

### Cost

**Benefits:** Free for GitHub Pro+ repositories, no additional tooling costs, no OIDC licensing requirements  
**Trade-offs:** Storage costs for artifacts (~minimal), potential duplicate infrastructure costs for int environment

### Technical Impact

**Benefits:** Aligns with existing .http testing pattern, reuses bash scripts, native Terraform integration, standard Azure authentication (widely understood)  
**Trade-offs:** GitHub Actions-specific syntax, manual secret rotation required (can be automated), secrets must be updated when rotated

### Team & Operations

**Benefits:** Self-service deployments, faster feedback loops, reduced manual toil, works on fresh subscriptions without prior setup, no local development Azure access required, simpler setup than OIDC  
**Trade-offs:** Secret rotation overhead (quarterly recommended), requires repo admin access for initial setup and rotation, setup user needs temporary elevated permissions

### Compliance & Security

**Benefits:** Built-in approval gates, deployment history, branch protection rules, secret rotation audit trail  
**Trade-offs:** Must configure environments correctly (no approval on int, approval on prod), must establish and enforce secret rotation policy

## Alternatives Considered

### Azure DevOps Pipelines

**Description:** Microsoft-native CI/CD platform with Azure integration

**How It Addresses Decision Drivers:**

- **Security:** Supports OIDC, has approval gates, strong Azure integration
- **Operational Efficiency:** Powerful but complex UI, more features than needed
- **Technical Fit:** Strong Terraform support, requires separate platform
- **Team Capabilities:** New tool to learn, separate from source control

**Rejection Rationale:** Code and pipelines live in different systems (split context), higher operational overhead, team already uses GitHub

### GitHub Actions with OIDC (Federated Credentials)

**Description:** Keyless authentication using OpenID Connect federated credentials

**How It Addresses Decision Drivers:**

- **Security:** No stored secrets, automatic token rotation, short-lived tokens (1 hour)
- **Technical Fit:** Native GitHub integration, Azure AD federated credentials
- **Team Capabilities:** More complex setup, federated credential configuration
- **Compliance:** Requires GitHub Enterprise Cloud or specific GitHub Enterprise Server versions

**Rejection Rationale:** Not available in all customer GitHub Enterprise environments (especially on-premises), adds complexity for marginal security benefit in CI/CD-only scenario, customer compatibility is critical requirement

### Jenkins with Azure Plugins

**Description:** Self-hosted automation server with Azure authentication plugins

**How It Addresses Decision Drivers:**

- **Security:** Requires service principal with secrets or certificate auth
- **Cost:** Infrastructure hosting costs, maintenance burden
- **Technical Fit:** Flexible but requires plugin management
- **Team Capabilities:** Requires Jenkins administration expertise, higher maintenance

**Rejection Rationale:** Self-hosting burden, long-lived credentials, no built-in environment concepts, overkill for project scope

### Terraform Cloud

**Description:** HashiCorp's managed Terraform execution platform

**How It Addresses Decision Drivers:**

- **Security:** VCS-driven workflow, supports OIDC, approval workflows
- **Cost:** Free tier limited, paid plans for team features
- **Technical Fit:** Excellent Terraform integration, requires webhook setup
- **Team Capabilities:** New platform to learn, separate UI

**Rejection Rationale:** Additional cost for team features, split deployment platform from code repository, HTTPYac testing still requires separate CI

## References

- **REF-001:** [GitHub Environments Documentation](https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment)
- **REF-002:** [Azure Service Principal Authentication](https://learn.microsoft.com/en-us/azure/developer/github/connect-from-azure)
- **REF-003:** [HTTPYac CLI Documentation](https://httpyac.github.io/guide/installation_cli.html)
- **REF-004:** [GitHub Secrets Management](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
