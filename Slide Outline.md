# Requirements Pipeline Presentation - Slide Outline

## Audience Context

- **Primary**: Your boss (awareness of work completed)
- **Secondary**: Digital Engineering IRAD Manager (evaluating AI enablement potential)

---

## Slide 1: Title

**Requirements Coverage Intelligence Pipeline**

- Subtitle: Leveraging AI for Requirements-to-Activities Matching
- Your name, Date
- Note: Focus on "what we built and demonstrated"

---

## Slide 2: Problem Statement

**The Requirements Coverage Gap**

**Current State:**

- Cameo model activities developed by systems engineers
- Requirements in JAMA developed by requirements team
- Manual matching process taking 2-4 hours per 100 activities
- No systematic way to identify coverage gaps

**Impact:**

- ~60% activities without clear requirement traces
- CDR readiness risk
- No visibility for phase leads on coverage metrics

---

## Slide 3: What I Built - System Overview

**Two Operational Capabilities:**

1. **Automated Matching Pipeline**
    - Processes CSV exports from JAMA and Cameo
    - Generates match scores for all requirement-activity pairs
    - Provides explanations for suggested matches
2. **Requirements Quality Analyzer**
    - Evaluates requirements against INCOSE patterns
    - Identifies specific quality issues
    - Grades requirements A-F with actionable feedback

**Note**: Both capabilities are fully implemented and tested on real data

---

## Slide 4: Technical Architecture

**Suggested Diagram**: Simple flow diagram showing: