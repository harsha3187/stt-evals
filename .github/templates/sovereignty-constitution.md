# UAE Cloud & AI Projects Constitution

## Core Principles

### I. Data Classification First (NON-NEGOTIABLE)

**Every** project MUST classify all data according to the four-tier classification framework **before** any implementation begins:

- **Open**: Publicly accessible data with no confidentiality requirements
- **Confidential/Restricted**: Data restricted to authorized personnel; unauthorized access could harm entities or individuals
- **Secret**: Data requiring high protection; unauthorized disclosure causes significant harm at national, Emirate, or sectoral level
- **Top Secret**: Highly restricted data; unauthorized disclosure threatens national security or organizational integrity

**Requirements**:

- Data classification MUST be documented in the project specification before Phase 0
- All design decisions, architecture choices, and security controls MUST align with the highest classification level of data processed
- Data classification MUST be reviewed and validated by security team before implementation
- Mixed classification levels within a single system MUST implement proper segmentation and access controls

**Rationale**: The UAE National Cloud Security Policy mandates data classification as the foundation for all security and sovereignty controls. Without proper classification, appropriate security measures cannot be determined.

### II. Sovereignty-Aware Architecture

Cloud and AI systems MUST implement sovereignty controls based on data classification:

**For Confidential/Restricted or higher data classifications**:

- Data storage MUST be within UAE geographical boundaries and legal jurisdiction
- Data processing MUST occur within UAE borders
- Encryption keys MUST be generated, stored, and managed within the UAE
- Foreign access requests MUST be denied by default
- All metadata storage and processing MUST remain within UAE jurisdiction

**For all classifications**:

- Data location awareness MUST be maintained at all times
- Data sovereignty compliance MUST be verifiable through audit trails
- Cross-border data transfer MUST be explicitly approved and documented
- CSP compliance with sovereignty requirements MUST be validated

**Rationale**: Data sovereignty is a fundamental requirement of the UAE National Cloud Security Policy, ensuring UAE maintains exclusive jurisdiction over its data.

### III. Risk-Based Security Approach

Security controls MUST be implemented based on documented risk assessments:

- Risk assessment MUST be performed before cloud adoption and at each major change
- Risk evaluation MUST consider data classification, business impact, and threat landscape
- Security controls MUST be proportionate to identified risks
- Risk management decisions MUST be documented and approved by appropriate authority
- Residual risks MUST be explicitly accepted by designated risk owner

**Security Control Tiers**:

- **Baseline Secure**: Minimum security standard for all projects (any classification)
- **Partial Sovereign**: Additional controls for Confidential/Restricted data
- **Sovereign**: Stringent controls for Secret and Top Secret data

**Rationale**: Risk-based approach is a core principle of the UAE Cloud Security Policy, ensuring resources are allocated efficiently to address the most significant threats.

### IV. Zero Trust Security Model

All systems MUST implement zero trust principles:

- Never trust, always verify - no implicit trust based on network location
- Least privilege access - users and services receive minimum necessary permissions
- Assume breach - design systems expecting compromise and limiting blast radius
- Continuous verification - authenticate and authorize every access request
- Micro-segmentation - segment systems to contain security breaches

**Implementation Requirements**:

- Multi-factor authentication (MFA) MUST be enforced for all access
- Role-based access control (RBAC) MUST be implemented with regular reviews
- All access attempts MUST be logged and monitored
- Privileged access MUST require additional authorization and monitoring
- Service-to-service communication MUST be authenticated and encrypted

**Rationale**: Zero trust architecture provides defense-in-depth essential for protecting sensitive government and critical infrastructure data.

### V. Privacy & Data Protection by Design

Privacy and data protection MUST be embedded from initial design:

- Personal Identifiable Information (PII) MUST be identified and protected
- Data minimization - collect only necessary data
- Purpose limitation - use data only for stated purposes
- Storage limitation - retain data only as long as necessary
- Data subject rights MUST be supported (access, correction, deletion)
- Privacy impact assessment MUST be conducted for systems processing PII

**PII Protection Requirements**:

- Direct identifiers (name, passport, Emirates ID, email, phone, biometrics, financial data, medical records) MUST be encrypted at rest and in transit
- Indirect identifiers (DOB, gender, location, employment data) MUST be protected based on aggregation risk
- PII MUST NOT be included in logs, error messages, or debugging output
- PII access MUST be logged and auditable

**Rationale**: Privacy protection is mandated by UAE data protection regulations and is essential for maintaining public trust in government digital services.

### VI. Continuous Compliance & Audit Readiness

Systems MUST be designed for continuous compliance monitoring and audit readiness:

**Minimum Certifications Required**:

- UAE Information Assurance Standard (self-assessment)
- ISO/IEC 27001:2022 (valid certificate)
- ISO/IEC 27017:2019 (cloud security - valid certificate)
- ISO/IEC 27018:2019 (PII protection in cloud - valid certificate)
- ISO/IEC 22301:2019 (business continuity - valid certificate)
- SOC 2 Type II (valid attestation report)
- CSA STAR 2 Certification/Attestation (valid)

**Audit & Compliance Requirements**:

- All security controls MUST be documentable and verifiable
- Compliance status MUST be monitored continuously with automated tooling where possible
- Audit trails MUST be tamper-proof and retained for minimum 10 years
- Non-compliance MUST be reported within defined timeframes
- Independent third-party assessments MUST be conducted annually
- Compliance evidence MUST be available within 24 hours of request

**Rationale**: Demonstrable compliance with UAE cyber security standards is mandatory for all government and critical infrastructure systems.

### VII. Incident Response & Resilience

Systems MUST be designed for rapid incident detection, response, and recovery:

**Incident Response Requirements**:

- Incident response plan MUST be documented and tested annually
- Security incidents MUST be reported to UAE Cyber Security Council per policy timelines:
  - Sovereign Cloud: Within 1 hour of confirmation
  - Partial Sovereign Cloud: Within 24 hours of confirmation
- Incident response team MUST be designated with defined roles and responsibilities
- All security events MUST be logged with sufficient detail for forensic analysis
- Forensic capabilities MUST support e-discovery and legal requirements

**Resilience Requirements**:

- Business Continuity Plan (BCP) MUST be documented and tested
- Disaster Recovery Plan (DRP) MUST define Recovery Time Objective (RTO) and Recovery Point Objective (RPO)
- Backups MUST be encrypted, tested regularly, and stored in geographically distinct locations within UAE
- Redundant infrastructure MUST be deployed for systems processing Secret or Top Secret data
- Failover testing MUST be conducted at least annually

**Rationale**: Rapid incident response and resilient systems are essential for maintaining continuity of critical government services and protecting national interests.

## Data Classification & Sovereignty Requirements

This section defines the relationship between data classification and required deployment models:

| Data Classification | Minimum CSP Tier | Data Storage Location | Processing Location | Key Management Location |
|---------------------|------------------|----------------------|---------------------|------------------------|
| Open | Baseline Secure | Any | Any | Any |
| Confidential/Restricted | Partial Sovereign | UAE only | UAE only | UAE only |
| Secret | Sovereign | UAE only | UAE only | UAE HSM only |
| Top Secret | Sovereign | UAE only | UAE only | UAE HSM only |

**Additional Sovereignty Controls for Secret/Top Secret**:

- CSP personnel MUST have UAE security clearances
- All data and metadata MUST remain under exclusive UAE jurisdiction
- Communication MUST route through UAE telecom infrastructure
- Environment MUST NOT be accessible from outside UAE borders
- Environment MUST NOT be accessible from lower-tier cloud environments
- On-site technical support MUST be available within UAE

## Security & Compliance Requirements

### Encryption & Cryptography

- Data at rest MUST be encrypted using approved algorithms (minimum AES-256)
- Data in transit MUST be encrypted using TLS 1.2 or higher
- Encryption keys for Confidential/Restricted+ data MUST be managed within UAE
- Hardware Security Modules (HSMs) MUST be used for Secret/Top Secret data
- Key rotation policies MUST be implemented and enforced
- Cryptographic implementations MUST be validated against FIPS 140-2 or equivalent

### Identity & Access Management

- Multi-factor authentication (MFA) MUST be enforced for all users
- Privileged access MUST require additional authentication
- Access rights MUST be reviewed quarterly
- User access MUST be removed within 24 hours of role change/termination
- Service accounts MUST use credential rotation
- Authentication events MUST be logged and monitored

### Application Security

- Secure development lifecycle (SDL) MUST be followed
- Code review MUST include security assessment
- Dependency vulnerabilities MUST be scanned and remediated
- Security testing MUST include SAST, DAST, and penetration testing
- Security vulnerabilities MUST be remediated per severity timelines:
  - Critical: 7 days
  - High: 30 days
  - Medium: 90 days
  - Low: 180 days

### Logging & Monitoring

- Security events MUST be logged with timestamp, user, action, and result
- Logs MUST be centrally collected and retained for minimum 1 year
- Real-time monitoring MUST detect suspicious activities
- Log integrity MUST be protected against tampering
- Logs MUST be reviewed regularly by security team

### Change Management

- All changes MUST follow documented change control process
- Security impact MUST be assessed for all changes
- Changes to production MUST require approval
- Rollback procedures MUST be documented and tested
- Emergency changes MUST be reviewed post-implementation

### Third-Party & Supply Chain Security

- Third-party security assessments MUST be conducted before engagement
- Vendor security compliance MUST be validated annually
- Data sharing agreements MUST define security requirements
- Vendor access MUST be monitored and logged
- Vendor security incidents MUST be reported

## Governance

### Constitutional Authority

This constitution is the supreme governance document for all UAE Cloud and AI projects. All development practices, architectural decisions, and implementation approaches MUST comply with these principles. In case of conflict between this constitution and other guidance, the constitution prevails.

### Amendment Process

Amendments to this constitution require:

1. **Proposal**: Document proposed change with rationale and impact analysis
2. **Review**: Security team and stakeholders review for 14 days minimum
3. **Approval**: CSC Governance or designated authority approves
4. **Version Update**: Semantic versioning (MAJOR.MINOR.PATCH):
   - MAJOR: Backward-incompatible governance changes, principle removal/redefinition
   - MINOR: New principle/section added, material guidance expansion
   - PATCH: Clarifications, wording improvements, non-semantic refinements
5. **Migration Plan**: For breaking changes, provide migration timeline and support
6. **Communication**: Notify all stakeholders within 5 business days
7. **Training**: Provide training materials for significant changes

### Compliance Verification

All projects MUST demonstrate constitutional compliance:

- **Phase 0 (Planning)**: Constitution check gate MUST pass before research begins
- **Phase 1 (Design)**: Re-verify constitutional compliance after design completion
- **Phase 2 (Implementation)**: Code reviews MUST verify compliance
- **Phase 3 (Deployment)**: Security assessment MUST confirm compliance
- **Ongoing**: Annual compliance audits MUST be conducted

Non-compliance MUST be:

- Documented with justification and risk acceptance
- Approved by appropriate authority (see Exception Handling)
- Remediated within defined timeline
- Tracked until resolution

### Exception Handling

Exceptions to constitutional requirements may be granted under special circumstances:

- Exception request MUST document: requirement, business justification, risk assessment, compensating controls, remediation plan
- Exceptions are reviewed by CSC or designated authority on case-by-case basis
- Exception approval is not guaranteed
- Approved exceptions MUST be time-limited (maximum 12 months)
- Exception status MUST be reviewed quarterly
- Remediation plan MUST be tracked to completion

### Performance Monitoring

Constitutional compliance and security posture MUST be monitored continuously:

- Security metrics MUST be defined and tracked
- Compliance dashboard MUST be maintained
- Trend analysis MUST identify systemic issues
- Regular reports MUST be provided to leadership
- Continuous improvement initiatives MUST address identified gaps

### Integration with UAE Initiatives

All projects MUST integrate and align with ongoing UAE cyber security initiatives:

- UAE National Cyber Security Strategy
- UAE National Cyber Security Governance Framework
- UAE National Cyber Security Incident Response Framework
- Critical Information Infrastructure Protection (CIIP) requirements
- Emirate-specific cyber security programs (where applicable)
- Telecommunications and Digital Government Regulatory Authority (TDRA) regulations

### Document References

This constitution is based on:

- UAE National Cloud Security Policy Version 2.0 (September 2025)
- UAE Information Assurance Standard
- International standards: ISO/IEC 27001:2022, 27017:2019, 27018:2019, 22301:2019

### Review Cycle

This constitution MUST be reviewed:

- Annually at minimum
- When UAE National Cloud Security Policy is updated
- When significant security threats emerge
- When technology landscape changes materially
- Upon request from CSC or designated authority

**Version**: 1.0.0 | **Ratified**: 2025-10-15 | **Last Amended**: 2025-10-15
