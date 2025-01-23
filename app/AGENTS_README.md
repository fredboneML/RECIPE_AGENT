# Agents Overview

This document provides a detailed overview of the agents implemented in the `ai_analyzer` project. Each agent plays a specific role in the call transcription analysis workflow and contributes to providing insightful data analysis for managers.

---

## **DatabaseInspectorAgent**

### Responsibilities:
- Inspects the database structure to understand the available data.
- Identifies available tables and their schemas.
- Configures and handles restrictions for tables to ensure security compliance.

---

## **QuestionGeneratorAgent**

### Responsibilities:
- Generates relevant analysis questions based on the database structure and available data.
- Focuses on key areas such as:
  - Sentiment trends.
  - Topics discussed during calls.
  - Customer satisfaction metrics.
- Designed to be model-agnostic, allowing for flexibility in integration with different machine learning models or tools.

---

## **SQLGeneratorAgent**

### Responsibilities:
- Converts natural language questions into SQL queries.
- Validates generated queries to ensure compliance with restrictions on specific tables or columns.
- Supports a configurable base context for improved query generation and accuracy.

---

## **ResponseAnalyzerAgent**

### Responsibilities:
- Analyzes responses from executed SQL queries or identifies errors in query execution.
- Suggests follow-up questions to refine insights or avoid errors.
- Summarizes query results into natural language for improved comprehension by managers.

---

## **CallAnalysisWorkflow**

### Responsibilities:
- Coordinates the interactions between all agents.
- Manages the overall workflow to ensure smooth execution of:
  - Database inspection.
  - Question generation.
  - Query translation and execution.
  - Response analysis.
- Provides a seamless experience for extracting insights from call transcription data.

---

This modular architecture ensures scalability and adaptability, enabling efficient and secure call transcription analysis tailored to business needs.
