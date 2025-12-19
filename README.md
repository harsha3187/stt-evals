# CAP MVE 

Multi-agent AI system built with Microsoft Agent Framework, featuring a Python FastAPI backend and Next.js frontend. Demonstrates modern cloud-native development with AI agents for clinical assistance, Arabic translation, FHIR data visualization, and web search.

## Documentation

0. [Project overview](docs/0-project-overview.md) - Architecture, tech stack, and folder structure
1. [Getting started](docs/1-getting-started.md) - Set up your local development environment
2. [Infrastructure setup](docs/2-infra.md) - Deploy Azure resources with Terraform
3. [Evaluations](docs/5-evals.md) - Testing and quality metrics
4. [GitHub Actions CD setup](docs/8-github-cd-setup.md) - Configure GitHub Actions for CI/CD

ADRs and service-specific documentation: 

- [ADRs (architecture decision records)](docs/adrs/)
- [Patient profile tool](docs/patient-profile-tool/)

## Quick start

Clone and open in DevContainer

```bash
git clone https://github.com/commercial-software-engineering/oryx-cap-upskilling.git
```

Open in VS Code → "Reopen in Container"

Install dependencies and run
```
make restore
make run-api  
```

The backend should be available at http://localhost:8000 


See [Getting started](docs/1-getting-started.md) for detailed setup instructions.


<!-----------------------[  Support & Reuse Expectations  ]-----<recommended> section below-------------->

### Support & Reuse Expectations

_The creators of this repository **DO NOT EXPECT REUSE**._

If you do use it, please let us know via an email or leave a note in an issue, so we can best understand the value of this repository.

<!-----------------------[  Links to Platform Policies  ]-------<recommended> section below-------------->

## How to Accomplish Common User Actions in GitHub inside Microsoft (GiM)
<!-- 
INSTRUCTIONS:
- This section links to information useful to any user of this repository new to internal GitHub policies & workflows.
-->

If you have trouble doing something related to this repository, please keep in mind that the following actions require using [GitHub inside Microsoft (GiM) tooling](https://aka.ms/gim/docs) and not the normal GitHub visible user interface!

- [Switching between EMU GitHub and normal GitHub without logging out and back in constantly](https://aka.ms/StartRight/README-Template/maintainingMultipleAccount)
- [Creating a repository](https://aka.ms/StartRight)
- [Changing repository visibility](https://aka.ms/StartRight/README-Template/policies/jit)
- [Gaining repository permissions, access, and roles](https://aka.ms/StartRight/README-TEmplates/gim/policies/access)
- [Enabling easy access to your low sensitivity and widely applicable repository by setting it to Internal Visibility and having any FTE who wants to see it join the 1ES Enterprise Visibility MyAccess Group](https://aka.ms/StartRight/README-Template/gim/innersource-access)
- [Migrating repositories](https://aka.ms/StartRight/README-Template/troubleshoot/migration)
- [Setting branch protection](https://aka.ms/StartRight/README-Template/gim/policies/branch-protection)
- [Setting up GitHubActions](https://aka.ms/StartRight/README-Template/policies/actions)
- [and other actions](https://aka.ms/StartRight/README-Template/gim/policies)

This README started as a template provided as part of the [StartRight](https://aka.ms/gim/docs/startright) tool that is used to create new repositories safely. Feedback on the [README template](https://aka.ms/StartRight/README-Template) used in this repository is requested as an issue.

<!-- version: 2023-04-07 [Do not delete this line, it is used for analytics that drive template improvements] -->
