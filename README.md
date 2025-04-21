# begone
12000 lines of Pydantic AI, Tavily, Streamlit just to do one thing really well - get rid of useless contents that are scraped.

![ParselyFi Demo](assets/ParselyFi%20Company%20Information%20Page%20v031825.gif)

## Core Challenge: Navigating Web Content Complexity

Automated company research relies heavily on information retrieved from the web. However, the modern web presents significant challenges beyond simple data retrieval:

1.  **Irrelevant Content & Noise:** Search results often include:
    *   Pages blocked by **bot protection mechanisms** (like Cloudflare, CAPTCHA challenges).
    *   **Paywalls** restricting access to full content.
    *   **Error pages** (404s, server errors, access denied).
    *   **Low-quality or generic content** (SEO articles, directories, templates).
    *   Content describing the *source's* services rather than the target company.

2.  **Entity Ambiguity:** Many companies share similar names. A key challenge is ensuring that retrieved information pertains to the *correct* target entity and not a different company with a similar name.

3.  **Data Accuracy & Consistency:** Information across different sources can be inconsistent or outdated.

**Our Multi-Layered Validation Approach:**

This project dedicates significant complexity to addressing these challenges through a multi-layered, AI-driven validation process:

*   **Content Validation (`content_validation_agent`):** An initial check uses an AI agent to classify retrieved page content, filtering out obvious non-content like security blocks, paywalls, and error pages *before* attempting deeper extraction.
*   **Entity Differentiation (`entity_verification_agent`, `enhanced_entity_differentiation_agent`):** When multiple potential company entities are found, AI agents compare detailed attributes (founders, founding year, location, business focus) extracted using `extract_entity_attributes` to determine if they represent the same or different companies. An `EntityDatabase` stores verified relationships.
*   **Negative Example Identification (`enhanced_negative_example_agent`):** Actively identifies and stores features of entities confirmed *not* to be the target company, helping to filter out irrelevant results mentioning these known non-matches.
*   **Strict Semantic Validation (`strict_semantic_validation_agent`):** Assesses whether a validated piece of content is truly relevant and consistent with the known profile of the target company, leveraging the differentiation results and negative examples.
*   **Field-Level Validation (`field_validation_agent`):** Performs a final check on extracted data points to ensure they are semantically valid for the specific field (e.g., ensuring a "Product Name" isn't just a generic category).

By employing these validation steps using specialized Pydantic AI agents powered by Gemini models, the tool aims to significantly improve the signal-to-noise ratio and deliver more accurate, reliable, and structured company information despite the inherent complexities of web data. While some systems use AI to trap bots in useless content (like Cloudflare's AI Labyrinth), this project uses AI to rigorously **filter out** irrelevant and inaccurate content *after* retrieval to ensure the quality of the final research data.
