# Data Model: Physical AI & Humanoid Robotics Textbook

## Core Entities

### Chapter
- **name**: String (e.g., "Introduction to Physical AI and Humanoid Robotics")
- **number**: Integer (1-15)
- **prerequisites**: List of Chapter numbers
- **components**: Map of required components (intro, technical_explanation, diagrams, examples, code_snippets, summary, quiz)
- **learning_objectives**: List of String
- **content**: Markdown content for each component
- **status**: Enum (draft, review, validated, published)

### CodeExample
- **id**: String (unique identifier within chapter)
- **chapter_id**: String (reference to parent chapter)
- **technology**: Enum (ros2, gazebo, isaac, python, c++)
- **description**: String
- **code**: String (actual code content)
- **framework_version**: String (e.g., "Humble Hawksbill", "Garden")
- **expected_output**: String
- **validation_status**: Enum (untested, passed, failed)
- **dependencies**: List of String (required packages or setups)

### Diagram
- **id**: String (unique identifier)
- **chapter_id**: String (reference to parent chapter)
- **title**: String
- **description**: String
- **type**: Enum (architecture, process, concept, workflow)
- **file_path**: String (path to SVG/PNG asset)
- **caption**: String
- **alt_text**: String (for accessibility)

### Quiz
- **id**: String (unique identifier within chapter)
- **chapter_id**: String (reference to parent chapter)
- **questions**: List of Question objects
- **passing_score**: Integer (percentage required to pass)

### Question
- **id**: String (unique identifier)
- **quiz_id**: String (reference to parent quiz)
- **type**: Enum (multiple_choice, practical_application, code_analysis, conceptual)
- **content**: String
- **options**: List of String (for multiple choice)
- **correct_answer**: String or Integer (index for multiple choice)
- **explanation**: String (why this is correct)

### GlossaryTerm
- **term**: String (the technical term)
- **definition**: String
- **category**: Enum (ros2, gazebo, isaac, humanoid, vla, general_robotics)
- **related_terms**: List of String
- **first_appears_in**: Integer (chapter number)

### HardwareReference
- **id**: String (unique identifier)
- **name**: String (product/model name)
- **category**: Enum (computing, sensor, actuator, platform, communication)
- **specifications**: Map of key-value pairs
- **compatibility**: Map of compatible systems (ros2, isaac, etc.)
- **use_case**: String
- **pros**: List of String
- **cons**: List of String
- **cost_range**: String
- **installation_requirements**: List of String

### Appendix
- **id**: String (appendix identifier A-G)
- **title**: String
- **content**: Markdown
- **references**: List of related chapters or concepts

## Relationships

### Chapter Relationships
- Each Chapter can have multiple Prerequisites (other Chapters)
- Each Chapter contains multiple CodeExample objects
- Each Chapter contains multiple Diagram objects
- Each Chapter has exactly one Quiz object
- Each Chapter references multiple GlossaryTerm objects

### CodeExample Relationships
- Each CodeExample belongs to exactly one Chapter
- Each CodeExample may reference multiple other CodeExample objects for dependencies

### Diagram Relationships
- Each Diagram belongs to exactly one Chapter
- Diagrams may reference other Diagrams for complex system views

### Quiz Relationships
- Each Quiz belongs to exactly one Chapter
- Each Quiz contains multiple Question objects
- Questions may reference specific CodeExample or Diagram objects

### GlossaryTerm Relationships
- GlossaryTerm objects are referenced by multiple Chapters
- GlossaryTerm objects may have relationships with other GlossaryTerm objects (related_terms)

## Validation Rules

### Chapter Validation
- Each chapter must contain all required components (intro, technical_explanation, diagrams, examples, code_snippets, summary, quiz)
- Chapter number must be between 1 and 15
- Prerequisites must reference existing chapters with lower numbers
- Learning objectives must align with the chapter content

### CodeExample Validation
- Code must be syntactically valid for the specified technology
- Framework version must be specified and valid
- Dependencies must be resolvable
- Expected output must be documented

### Diagram Validation
- File path must exist and be accessible
- Alt text must be provided for accessibility
- Caption must be descriptive and relevant

### Quiz Validation
- At least 5 questions per quiz
- Mix of question types (at least 2 multiple choice, 1 practical application, 1 code analysis, 1 conceptual)
- Passing score must be between 70-90%

### GlossaryTerm Validation
- Term must be used in at least one chapter
- Definition must be clear and concise
- Category must be specified

## State Transitions

### Chapter States
- draft → review (when initial content is complete)
- review → validated (when reviewed and approved)
- validated → published (when ready for release)

### CodeExample States
- untested → passed (when code executes successfully)
- untested → failed (when code has errors)
- failed → passed (when errors are fixed and re-tested)

### Diagram States
- created → reviewed (when reviewed for accuracy)
- reviewed → approved (when approved for use)